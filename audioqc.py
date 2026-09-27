"""Clip analysis (silence, pauses, clipping, loudness, speaking rate) and cleaning."""
import io
import math
import wave

import numpy as np
import regex as re

FLAG_TEXT = {
    "silent": "No speech found", "short": "Very short", "long": "Over 15 s", "cutoff": "Ends abruptly",
    "lead": "Long silence before speech", "pause": "Long pause mid-sentence", "clip": "Clipping",
    "quiet": "Very quiet", "fast": "Too fast for its text", "slow": "Too slow for its text",
    "asr": "Transcript mismatch",
}


def pcm16_to_f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(dst, f, sr):
    i16 = (np.clip(f, -1, 1) * 32767).round().astype("<i2")
    with wave.open(dst if not hasattr(dst, "__fspath__") else str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(i16.tobytes())


def wav_bytes(f, sr) -> bytes:
    buf = io.BytesIO()
    write_wav(buf, f, sr)
    return buf.getvalue()


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM")
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def analyze(f: np.ndarray, sr: int, text: str, c: dict) -> dict:
    n = len(f)
    hop = int(round(sr * 0.01))
    nF = max(1, n // hop)
    fr = np.zeros(nF * hop, dtype=np.float64)
    m = min(n, nF * hop)
    fr[:m] = f[:m]
    db = 10 * np.log10((fr.reshape(nF, hop) ** 2).mean(1) + 1e-12)
    a = np.abs(f)
    peak = float(a.max()) if n else 0.0
    clip = int((a >= 0.999).sum())
    thr = max(c["sil_db"], float(db.max()) - 40)
    v = db > thr
    cs = np.concatenate([[0], np.cumsum(v)])
    idx = np.arange(nF)
    fw = cs[np.minimum(nF, idx + 5)] - cs[idx]
    bw = cs[idx + 1] - cs[np.maximum(0, idx - 4)]
    fs = np.where(v & (fw >= 3))[0]
    ls = np.where(v & (bw >= 3))[0]
    dur = n / sr
    if not len(fs) or not len(ls) or ls[-1] < fs[0]:
        return {"dur": dur, "speech": 0, "lead": dur, "trail": 0, "max_pause": 0, "peak": peak, "clip": clip,
                "rms_db": -120, "rate": 0, "trim_start": 0, "trim_end": n, "gain": 1.0, "out_dur": dur,
                "flags": ["silent"], "z": None}
    first, last = int(fs[0]), int(ls[-1])
    seg = v[first:last + 1]
    run = mx = 0
    for x in seg:
        run = 0 if x else run + 1
        mx = max(mx, run)
    rms_db = 10 * math.log10(float(np.mean(10 ** (db[first:last + 1][seg] / 10))) + 1e-12)
    speech = (last - first + 1) * hop / sr
    lead = first * hop / sr
    trail = max(0, n - (last + 1) * hop) / sr
    max_pause = mx * hop / sr
    letters = len(re.findall(r"[\p{L}\p{M}]", text))
    rate = letters / max(speech, 0.1)
    ts = int(round(first * hop - c["pad_lead"] * sr))
    te = int(round((last + 1) * hop + c["pad_trail"] * sr))
    gain = min(10 ** ((c["target_rms"] - rms_db) / 20), 10 ** (c["peak_db"] / 20) / max(peak, 1e-6))
    out_dur = (te - ts) / sr
    flags = []
    if speech < 0.5: flags.append("short")
    if out_dur > 15: flags.append("long")
    if trail < 0.04: flags.append("cutoff")
    if lead > 1.2: flags.append("lead")
    if max_pause > c["max_pause"]: flags.append("pause")
    if clip > 8: flags.append("clip")
    if rms_db < c["min_rms"]: flags.append("quiet")
    return {"dur": dur, "speech": speech, "lead": lead, "trail": trail, "max_pause": max_pause, "peak": peak,
            "clip": clip, "rms_db": rms_db, "rate": rate, "trim_start": ts, "trim_end": te, "gain": gain,
            "out_dur": out_dur, "flags": flags, "z": None}


def clean(f: np.ndarray, q: dict, sr: int) -> np.ndarray:
    """Trim with fixed padding (adds digital silence if needed), remove DC, normalise, short fades."""
    s, e = q["trim_start"], q["trim_end"]
    out = np.zeros(max(0, e - s), dtype=np.float32)
    a, b = max(0, s), min(len(f), e)
    if b > a:
        seg = f[a:b] - f[a:b].mean()
        out[a - s:b - s] = seg * q["gain"]
    fl = min(len(out) // 2, int(0.005 * sr))
    if fl:
        ramp = np.linspace(0, 1, fl, dtype=np.float32)
        out[:fl] *= ramp
        out[-fl:] *= ramp[::-1]
    return np.clip(out, -1, 1)
