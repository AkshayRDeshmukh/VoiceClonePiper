"""Minimal ElevenLabs and OpenAI REST clients with retry/backoff."""
import json
import random
import time

import requests

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ApiError(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status

    @property
    def fatal(self):
        return self.status in (401, 402, 403)


def _err(r):
    msg = r.text[:300]
    try:
        j = r.json()
        if isinstance(j, dict) and "error" in j and isinstance(j["error"], dict):
            msg = j["error"].get("message", msg)
        elif isinstance(j, dict) and "detail" in j:
            d = j["detail"]
            msg = d.get("message") or d.get("status") or str(d) if isinstance(d, dict) else str(d)
    except Exception:
        pass
    return ApiError(f"{r.status_code}: {msg}", r.status_code)


def with_retry(fn, tries=6, log=None, should_stop=None):
    for a in range(tries):
        try:
            return fn()
        except ApiError as e:
            if e.status not in RETRY_STATUS or a == tries - 1:
                raise
            reason = str(e)
        except requests.RequestException as e:
            if a == tries - 1:
                raise ApiError(f"network error: {e}")
            reason = f"network error: {e}"
        wait = min(40, 1.5 * 2 ** a) + random.random()
        if log:
            log(f"{reason[:120]} — retrying in {wait:.0f}s", "warn")
        if should_stop and should_stop():
            raise ApiError("paused", 0)
        time.sleep(wait)


class ElevenLabs:
    BASE = "https://api.elevenlabs.io"

    def __init__(self, key):
        self.key = key
        self.s = requests.Session()

    def _req(self, method, path, timeout=120, headers=None, **kw):
        h = {"xi-api-key": self.key, **(headers or {})}
        r = self.s.request(method, self.BASE + path, headers=h, timeout=timeout, **kw)
        if not r.ok:
            raise _err(r)
        return r

    def subscription(self):
        return self._req("GET", "/v1/user/subscription").json()

    def voices(self):
        return self._req("GET", "/v1/voices").json().get("voices", [])

    def tts(self, text, voice_id, model_id, settings, seed, sample_rate=22050, lang=None):
        body = {"text": text, "model_id": model_id, "seed": int(seed) % 4294967295, "voice_settings": settings}
        if lang:
            body["language_code"] = lang
        r = self._req("POST", f"/v1/text-to-speech/{voice_id}", params={"output_format": f"pcm_{sample_rate}"},
                      json=body, headers={"Accept": "audio/*"})
        b = r.content
        return b[: len(b) // 2 * 2]

    def stt(self, wav, model_id, lang=None):
        data = {"model_id": model_id, "tag_audio_events": "false"}
        if lang:
            data["language_code"] = lang
        r = self._req("POST", "/v1/speech-to-text", data=data, files={"file": ("clip.wav", wav, "audio/wav")})
        return r.json().get("text", "")


class OpenAI:
    BASE = "https://api.openai.com/v1"

    def __init__(self, key):
        self.key = key
        self.s = requests.Session()

    def _req(self, method, path, timeout=180, **kw):
        r = self.s.request(method, self.BASE + path, headers={"Authorization": f"Bearer {self.key}"},
                           timeout=timeout, **kw)
        if not r.ok:
            raise _err(r)
        return r

    def check_model(self, model):
        return self._req("GET", f"/models/{model}").json()

    def chat_json(self, model, system, user):
        r = self._req("POST", "/chat/completions", json={
            "model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).json()
        txt = r["choices"][0]["message"]["content"] or "{}"
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            raise ApiError("model returned invalid JSON", None)
        return data, int(r.get("usage", {}).get("total_tokens", 0))

    def transcribe(self, wav, model, lang=None):
        data = {"model": model}
        if lang:
            data["language"] = lang
        r = self._req("POST", "/audio/transcriptions", data=data, files={"file": ("clip.wav", wav, "audio/wav")})
        return r.json().get("text", "")
