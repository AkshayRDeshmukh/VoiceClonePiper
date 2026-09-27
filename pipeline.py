"""Voice Forge pipeline: resumable stages from sentence writing to a final Piper ONNX voice."""
import collections
import json
import os
import re as _re
import shlex
import shutil
import signal
import sqlite3
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path

import requests

import audioqc as aq
import textproc as tp
from apis import ApiError, ElevenLabs, OpenAI, with_retry

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RAW, REFS, EVAL, RUNS = WORK / "raw", WORK / "refs", WORK / "eval", WORK / "runs"
DS = WORK / "dataset"
DSWAV = DS / "wav"
OUT = ROOT / "output"
CKPT_DIR = ROOT / "checkpoints"
PIPER_DIR = ROOT / "third_party" / "piper1-gpl"
PIPER_PY = PIPER_DIR / ".venv" / "bin" / "python"
CONFIG = WORK / "config.json"
BASE_CKPT = CKPT_DIR / "base-medium.ckpt"

LESSAC = ("https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
          "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt")

DEFAULT_CONFIG = {
    "elevenlabs_api_key": "", "openai_api_key": "",
    "voice_id": "", "voice_label": "", "el_model": "eleven_multilingual_v2", "language_code": "en",
    "stability": 0.6, "similarity": 0.8, "style": 0.0, "speed": 1.0, "speaker_boost": True, "seed": 424242,
    "sample_rate": 22050,
    "text_language": "English", "domain": "", "openai_model": "gpt-5-mini", "test_sentences": 30,
    "pool_factor": 2.0, "min_chars": 20, "max_chars": 180,
    "target_minutes": 90, "stt_provider": "elevenlabs", "stt_model": "scribe_v2", "wer_threshold": 0.15,
    "max_attempts": 3, "rate_z": 3.0, "concurrency": 3, "max_el_credits": 0,
    "sil_db": -45, "pad_lead": 0.15, "pad_trail": 0.25, "max_pause": 0.9, "target_rms": -21, "peak_db": -1,
    "min_rms": -38,
    "voice_name": "en_US-myvoice-medium", "espeak_voice": "en-us", "base_ckpt_url": LESSAC, "base_epoch": 2164,
    "epochs_first": 1000, "epochs_improve": 300, "max_rounds": 3, "target_wer": 0.05, "batch_size": 0,
    "extra_train_args": "", "train_device": "auto", "improve_words": 25, "improve_per_word": 6, "eval_with_apis": True,
    "expressive_minutes": 25, "expressive_stability": 0.35, "expressive_style": 0.3,
}

STAGES = [("check", "Check setup"), ("corpus", "Write sentences"), ("select", "Pick script"),
          ("synth", "Generate speech"), ("verify", "Transcribe and check"), ("repair", "Fix or reject"),
          ("export", "Build dataset"), ("train", "Train"), ("evaluate", "Test voice"),
          ("improve", "Improve dataset"), ("expressive", "Expressive round")]
STAGE_LABEL = dict(STAGES)

TOPICS = ["everyday home life", "cooking and food", "travel and directions", "weather and seasons",
          "technology and gadgets", "health and fitness", "money and banking", "sports", "science facts",
          "history", "nature and animals", "city life and transport", "feelings and relationships",
          "customer service", "news headlines", "a mystery novel", "a children's story", "work and meetings",
          "school and learning", "music and film", "shopping", "family", "farming and villages",
          "space and astronomy", "the ocean", "festivals and celebrations", "law and government",
          "medicine and hospitals", "smart home and voice assistants", "cars and roads", "art and museums",
          "business and startups", "friendship", "hobbies and crafts", "geography", "phone calls",
          "restaurants", "airports and trains", "emergencies and safety", "philosophy and ideas"]
STYLES = ["plain statements", "questions people ask", "excited, surprised or urgent exclamations",
          "lines someone says out loud in conversation", "narration from a novel",
          "instructions and directions", "polite service phrases", "news-reading style"]
LENGTHS = ["short, four to eight words", "medium, nine to eighteen words",
           "long, nineteen to thirty words with commas"]

EMOTIONS = ["joyful and delighted", "excited and amazed", "warm and tender", "playful and teasing",
            "surprised or shocked", "sad and disappointed", "worried or nervous", "frustrated or annoyed",
            "proud and triumphant", "calm and reassuring", "curious and wondering", "sarcastic and dry",
            "apologetic", "urgent and alarmed", "nostalgic and wistful", "encouraging and motivating"]
EXPR_FORMS = ["short spoken reactions", "lines of dialogue a character says out loud",
              "sentences from an emotional moment in a story", "things a friend says on the phone",
              "questions full of feeling", "exclamations"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sent(
  id INTEGER PRIMARY KEY AUTOINCREMENT, nkey TEXT UNIQUE, text TEXT, type TEXT, source TEXT,
  status TEXT, round INTEGER DEFAULT 1, attempts INTEGER DEFAULT 0, qc TEXT, dur REAL, raw TEXT,
  exported TEXT, wer REAL, asr TEXT, errwords TEXT, decision TEXT, note TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS ix_status ON sent(status);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""

SYSTEM_PROMPT = ("You write sentences for recording a text-to-speech training dataset. "
                 "Always reply with a single JSON object and nothing else.")
RULES = """Rules:
- Write every word exactly as it should be spoken: spell out all numbers, dates, times, money, units and abbreviations (for example "twenty-five percent", "Doctor", "half past three").
- No digits, no symbols such as % & $ # @ / + =, no emojis, no web addresses, no quotation marks, no parentheses, no all-capital acronyms.
- Natural and grammatical, with varied vocabulary: mix everyday and less common words, and use names of people and places now and then.
- Each sentence stands alone and ends with a full stop, question mark or exclamation mark.
- Avoid starting two sentences with the same words."""


class PipelineError(Exception):
    pass


class Handoff(Exception):
    """Dataset is ready but this machine can't train: pause cleanly with instructions."""
    pass


def utt(r):
    return f"utt_{r['id']:05d}"


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG.exists():
        try:
            cfg.update(json.loads(CONFIG.read_text()))
        except json.JSONDecodeError:
            pass
    return cfg


def save_config(cfg):
    WORK.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2))
    os.chmod(CONFIG, 0o600)


class Store:
    def __init__(self, path):
        self.c = sqlite3.connect(str(path), check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.c.executescript(SCHEMA)
            self.c.commit()

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.c.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock:
            cur = self.c.execute(sql, args)
            self.c.commit()
            return cur.rowcount

    def many(self, sql, rows):
        with self.lock:
            self.c.executemany(sql, rows)
            self.c.commit()

    def get(self, k, default=None):
        r = self.q("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(r[0]["v"]) if r else default

    def set(self, k, v):
        self.x("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    def incr(self, k, n):
        with self.lock:
            self.set(k, (self.get(k, 0) or 0) + n)

    def update(self, sid, **f):
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
        cols = ", ".join(f"{k}=?" for k in f)
        self.x(f"UPDATE sent SET {cols}, updated=? WHERE id=?", (*vals, time.time(), sid))

    def insert(self, items, status, source, rnd):
        """items: list of (text, type). Returns number actually inserted (duplicates skipped)."""
        n = 0
        with self.lock:
            for text, typ in items:
                cur = self.c.execute(
                    "INSERT OR IGNORE INTO sent(nkey,text,type,source,status,round,updated) VALUES(?,?,?,?,?,?,?)",
                    (tp.norm_key(text), text, typ, source, status, rnd, time.time()))
                n += cur.rowcount
            self.c.commit()
        return n

    def count(self, status):
        return self.q("SELECT COUNT(*) n FROM sent WHERE status=?", (status,))[0]["n"]


class Runner:
    def __init__(self):
        for d in (WORK, RAW, REFS, EVAL, RUNS, DSWAV, OUT, CKPT_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.db = Store(WORK / "project.db")
        self.cfg = load_config()
        self.logs = collections.deque(maxlen=3000)
        self.log_id = 0
        self.log_lock = threading.Lock()
        self.logfile = open(WORK / "pipeline.log", "a", buffering=1, encoding="utf-8")
        self.thread = None
        self.pause_flag = False
        self.status = "idle"
        self.message = "Add your keys and voice in Settings, then press Start."
        self.progress = {}
        self.train = {}
        self.proc = None
        try:  # show recent history after a restart
            for line in (WORK / "pipeline.log").read_text(encoding="utf-8").splitlines()[-300:]:
                m = _re.match(r"\S+ \S+ \[(\w+)\] (.*)", line)
                if m:
                    self.log_id += 1
                    self.logs.append({"id": self.log_id, "t": time.time(), "lvl": m.group(1), "msg": m.group(2)})
        except OSError:
            pass
        if self.db.get("stage") == "done":
            self.status, self.message = "done", f"Finished. Your voice is in {OUT / 'final'}"
        elif self.cfg["elevenlabs_api_key"] and self.cfg["openai_api_key"] and self.cfg["voice_id"]:
            self.message = "Ready. Press Start (or Resume) to continue the pipeline."

    # ---------- infra ----------
    def log(self, msg, lvl="info"):
        with self.log_lock:
            self.log_id += 1
            e = {"id": self.log_id, "t": time.time(), "lvl": lvl, "msg": str(msg)}
            self.logs.append(e)
            self.logfile.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{lvl}] {msg}\n")

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running:
            return
        self.pause_flag = False
        self.status = "running"
        self.message = ""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pause(self):
        if self.running:
            self.pause_flag = True
            self.status = "pausing"
            self.message = "Finishing current work, then pausing…"
            self._stop_proc()

    def goto(self, stage):
        if self.running:
            raise PipelineError("Pause the pipeline first.")
        if stage not in STAGE_LABEL:
            raise PipelineError("Unknown stage.")
        order = [k for k, _ in STAGES]
        rnd = self.db.get("round", 1)
        if order.index(stage) <= order.index("train"):
            # Re-running training: allow it to continue (from our own last checkpoint, if any) and
            # make sure the old ONNX of this round is rebuilt instead of reused.
            info = self.db.get(f"train_r{rnd}") or {}
            info["done"] = False
            info.pop("max_epochs", None)
            self.db.set(f"train_r{rnd}", info)
            for p in (OUT / f"round_{rnd}").glob("*.onnx*"):
                p.unlink()
            self.db.set(f"report_r{rnd}", None)
        self.db.set("stage", stage)
        self.status = "paused"
        self.message = f"Ready to run from: {STAGE_LABEL[stage]}. Press Resume."
        self.log(f"Next run will start at: {STAGE_LABEL[stage]}")

    def stopped(self):
        return self.pause_flag

    def set_progress(self, label, done, total, detail=""):
        p = self.progress
        if p.get("label") != label:
            p = self.progress = {"label": label, "t0": time.time(), "d0": done}
        p.update(done=done, total=total, detail=detail)
        rate_n = done - p["d0"]
        el = time.time() - p["t0"]
        p["eta"] = (el / rate_n) * (total - done) if rate_n > 0 and total else None

    def _stop_proc(self):
        pr = self.proc
        if pr and pr.poll() is None:
            try:
                os.killpg(pr.pid, signal.SIGINT)
            except Exception:
                pass

            def hard():
                time.sleep(90)
                if pr.poll() is None:
                    try:
                        os.killpg(pr.pid, signal.SIGKILL)
                    except Exception:
                        pass
            threading.Thread(target=hard, daemon=True).start()

    def _run(self):
        try:
            self.cfg = load_config()
            self.el = ElevenLabs(self.cfg["elevenlabs_api_key"])
            self.oa = OpenAI(self.cfg["openai_api_key"])
            while not self.pause_flag:
                st = self.db.get("stage", "check")
                if st == "done":
                    self.status = "done"
                    self.message = f"Finished. Your voice is in {OUT / 'final'}"
                    self.log(self.message)
                    return
                self.log(f"— {STAGE_LABEL[st]} —")
                self.progress = {}
                nxt = getattr(self, "st_" + st)()
                if nxt is None:
                    break
                self.db.set("stage", nxt)
            self.status = "paused"
            self.message = "Paused. Press Resume to continue where it stopped."
            self.log("Paused.")
        except Handoff as e:
            self.status = "paused"
            self.message = str(e)
            self.log(str(e), "warn")
        except (PipelineError, ApiError) as e:
            self.status = "error"
            self.message = str(e)
            self.log(str(e), "error")
        except Exception as e:
            self.status = "error"
            self.message = f"Unexpected error: {e}"
            self.log(traceback.format_exc(), "error")

    def run_parallel(self, items, fn, label, workers):
        it = iter(items)
        lock = threading.Lock()
        state = {"done": 0, "errs": 0, "fatal": None}
        total = len(items)
        self.set_progress(label, 0, total)

        def worker():
            while not self.pause_flag and state["fatal"] is None:
                with lock:
                    x = next(it, None)
                if x is None:
                    return
                try:
                    fn(x)
                except ApiError as e:
                    if e.fatal:
                        state["fatal"] = e
                    elif e.status != 0:
                        self.log(f"{utt(x)}: {e}", "warn")
                        state["errs"] += 1
                except Exception as e:
                    self.log(f"{utt(x)}: {e}", "warn")
                    state["errs"] += 1
                with lock:
                    state["done"] += 1
                    self.set_progress(label, state["done"], total,
                                      f"{state['errs']} errors" if state["errs"] else "")

        ths = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, int(workers)))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        if state["fatal"]:
            raise PipelineError(f"API stopped the run: {state['fatal']}")

    # ---------- helpers ----------
    @property
    def english(self):
        return str(self.cfg.get("language_code") or "en").lower().startswith("en")

    def qc_cfg(self):
        return {k: float(self.cfg[k]) for k in ("sil_db", "pad_lead", "pad_trail", "max_pause",
                                                 "target_rms", "peak_db", "min_rms")}

    def sr(self):
        return int(self.cfg["sample_rate"])

    def credit_factor(self):
        return 0.5 if _re.search(r"flash|turbo", self.cfg["el_model"]) else 1.0

    def target_sentences(self):
        return int(self.cfg["target_minutes"] * 60 / 5.0)

    def voice_settings(self, expressive=False):
        c = self.cfg
        return {"stability": float(c["expressive_stability"] if expressive else c["stability"]),
                "similarity_boost": float(c["similarity"]),
                "style": float(c["expressive_style"] if expressive else c["style"]),
                "use_speaker_boost": bool(c["speaker_boost"]), "speed": float(c["speed"])}

    def tts(self, text, seed, expressive=False):
        c = self.cfg
        vs = self.voice_settings(expressive)
        pcm = with_retry(lambda: self.el.tts(text, c["voice_id"], c["el_model"], vs, seed,
                                             self.sr(), c["language_code"] or None),
                         log=self.log, should_stop=self.stopped)
        self.db.incr("el_chars", len(text))
        return aq.pcm16_to_f32(pcm)

    def stt(self, f, sr):
        c = self.cfg
        wav = aq.wav_bytes(f, sr)
        lang = c["language_code"] or None
        if c["stt_provider"] == "openai":
            return with_retry(lambda: self.oa.transcribe(wav, c["stt_model"], lang), log=self.log,
                              should_stop=self.stopped)
        return with_retry(lambda: self.el.stt(wav, c["stt_model"], lang), log=self.log, should_stop=self.stopped)

    def ask_sentences(self, prompt):
        data, tokens = with_retry(lambda: self.oa.chat_json(self.cfg["openai_model"], SYSTEM_PROMPT, prompt),
                                  log=self.log, should_stop=self.stopped)
        self.db.incr("oa_tokens", tokens)
        out = data.get("sentences", []) if isinstance(data, dict) else []
        return [s for s in out if isinstance(s, str)]

    def clean_many(self, raw):
        good, rejects = [], collections.Counter()
        for s in raw:
            t, why = tp.clean_sentence(s, self.english, int(self.cfg["min_chars"]), int(self.cfg["max_chars"]))
            if why:
                rejects[why] += 1
            else:
                good.append((t, tp.sentence_type(t)))
        return good, rejects

    def problems(self, r):
        q = json.loads(r["qc"]) if r["qc"] else {}
        f = list(q.get("flags", []))
        z = q.get("z")
        zlim = float(self.cfg["rate_z"]) * (1.5 if r.get("source") == "openai-expressive" else 1.0)
        if z is not None and abs(z) > zlim:
            f.append("fast" if z > 0 else "slow")
        if r["wer"] is not None and r["wer"] > float(self.cfg["wer_threshold"]):
            f.append("asr")
        return f

    def latest_ckpt(self):
        """Newest checkpoint written by *our* training (prefers Lightning's last.ckpt)."""
        cks = list(RUNS.glob("lightning_logs/version_*/checkpoints/*.ckpt"))
        if not cks:
            return None
        newest_dir = max({p.parent for p in cks}, key=lambda d: max(x.stat().st_mtime for x in d.glob("*.ckpt")))
        last = newest_dir / "last.ckpt"
        return last if last.exists() else max(newest_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)

    def pick_resume(self):
        """Newest checkpoint of ours that actually loads (a full disk can leave a truncated file)."""
        cks = sorted(RUNS.glob("lightning_logs/version_*/checkpoints/*.ckpt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        for p in cks[:8]:
            r = subprocess.run([str(PIPER_PY), "-c", "import sys, torch; c = torch.load(sys.argv[1], "
                                "map_location='cpu', weights_only=False); assert 'optimizer_states' in c", str(p)],
                               capture_output=True, text=True, timeout=900, cwd=str(PIPER_DIR))
            if r.returncode == 0:
                return p
            self.log(f"Skipping unreadable checkpoint {p.parent.parent.name}/{p.name} "
                     "(probably cut off when the disk filled up).", "warn")
        if cks:
            self.log("No readable checkpoint of ours found; warm-starting from the base checkpoint again.", "warn")
        return None

    def prune_runs(self, keep, rnd):
        """Delete old run folders/checkpoints and stale caches. Keeps only the checkpoint we resume from."""
        freed = 0
        keep = Path(keep) if keep else None
        for vdir in RUNS.glob("lightning_logs/version_*"):
            if keep and keep.parent.parent == vdir:
                for p in vdir.glob("checkpoints/*.ckpt"):
                    if p != keep:
                        freed += p.stat().st_size
                        p.unlink()
                continue
            freed += sum(p.stat().st_size for p in vdir.rglob("*") if p.is_file())
            shutil.rmtree(vdir, ignore_errors=True)
        for d in WORK.glob("cache_r*"):
            if d.name != f"cache_r{rnd}":
                freed += sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                shutil.rmtree(d, ignore_errors=True)
        if self.db.get("device", {}).get("device") not in (None, "none"):
            for p in (OUT / "handoff").glob("*.tar.gz"):  # not needed when this machine trains
                freed += p.stat().st_size
                p.unlink()
        if freed > 50 * 2**20:
            self.log(f"Freed {freed / 2**30:.1f} GB of old checkpoints, caches and archives.")

    def piper_run(self, args, timeout=None):
        r = subprocess.run([str(PIPER_PY), *map(str, args)], capture_output=True, text=True, timeout=timeout,
                           cwd=str(PIPER_DIR))
        if r.returncode:
            raise PipelineError(f"{args[:3]} failed:\n{(r.stderr or r.stdout)[-1500:]}")
        return r.stdout

    # ---------- stages ----------
    def st_check(self):
        c = self.cfg
        for k, name in (("elevenlabs_api_key", "ElevenLabs API key"), ("openai_api_key", "OpenAI API key"),
                        ("voice_id", "ElevenLabs voice")):
            if not c[k]:
                raise PipelineError(f"Set the {name} in Settings.")
        sub = with_retry(self.el.subscription, log=self.log)
        left = int(sub.get("character_limit", 0)) - int(sub.get("character_count", 0))
        self.db.set("el_left", left)
        self.log(f"ElevenLabs OK: plan {sub.get('tier')}, {left:,} credits left.")
        with_retry(lambda: self.oa.check_model(c["openai_model"]), log=self.log)
        self.log(f"OpenAI OK: model {c['openai_model']} available.")
        dev = self.probe_device()
        if dev["device"] == "none":
            self.log("No usable training device here. Sentences, audio and the dataset will be prepared, "
                     "then the run pauses and packs the project so you can resume on a GPU machine.", "warn")
        if PIPER_PY.exists() and not BASE_CKPT.exists():
            self.download(c["base_ckpt_url"], BASE_CKPT)
        return "corpus"

    def probe_device(self):
        """Which device can train here: cuda / mps / cpu / none. Re-probed at training time."""
        info = {"device": "none", "name": "", "vram_mb": 0}
        if not PIPER_PY.exists():
            self.log("Piper trainer not installed on this machine.", "warn")
        else:
            chk = subprocess.run([str(PIPER_PY), "-c", "from piper import espeakbridge"], capture_output=True,
                                 text=True, cwd=str(PIPER_DIR))
            if chk.returncode:
                raise PipelineError("Piper's phonemizer (espeakbridge) isn't built. Rerun the setup script "
                                    "(./setup_mac.sh --with-trainer on a Mac, ./setup.sh on Linux), then press Resume.")
            out = self.piper_run(["-c", "import torch, piper.train\n"
                                        "c=torch.cuda.is_available()\n"
                                        "m=hasattr(torch.backends,'mps') and torch.backends.mps.is_available()\n"
                                        "print(c); print(m)\n"
                                        "print(torch.cuda.get_device_name(0) if c else 'none')\n"
                                        "print(torch.cuda.get_device_properties(0).total_memory//2**20 if c else 0)"],
                                 timeout=300).split("\n")
            cuda, mps = out[0].strip() == "True", out[1].strip() == "True"
            want = self.cfg.get("train_device", "auto")
            if cuda and want in ("auto", "cuda"):
                info = {"device": "cuda", "name": out[2].strip(), "vram_mb": int(out[3].strip() or 0)}
            elif mps and want == "mps":
                info = {"device": "mps", "name": "Apple GPU (experimental)", "vram_mb": 8192}
            elif want == "cpu":
                info = {"device": "cpu", "name": "CPU (very slow)", "vram_mb": 8192}
            self.log(f"Training device: {info['device']} {info['name']}".strip()
                     + (f", {info['vram_mb'] // 1024} GB" if info["device"] == "cuda" else "")
                     + (" — Apple GPU detected; set Training device to 'mps' to try it." if mps and want == "auto" else ""),
                     "info" if info["device"] != "none" else "warn")
        self.db.set("device", info)
        self.db.set("vram_mb", info["vram_mb"])
        return info

    def handoff(self):
        """Pack code + project state so it can be resumed on a GPU machine."""
        import tarfile
        dst = OUT / "handoff" / f"voiceforge-{self.cfg['voice_name']}.tar.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".html", ".txt", ".md")]
        work = [p for p in WORK.rglob("*") if p.is_file() and not any(
            part.startswith("cache_") or part == "runs" for part in p.relative_to(WORK).parts)]
        self.set_progress("Packing project for a GPU machine", 0, len(work))
        with tarfile.open(dst, "w:gz", compresslevel=3) as tar:
            for p in files:
                tar.add(p, f"voiceforge/{p.name}")
            for i, p in enumerate(work):
                tar.add(p, f"voiceforge/work/{p.relative_to(WORK)}")
                if i % 200 == 0:
                    self.set_progress("Packing project for a GPU machine", i, len(work))
        size = dst.stat().st_size / 2**20
        raise Handoff(f"Dataset ready. This machine can't train, so the project was packed to {dst} ({size:.0f} MB). "
                      "Copy it to a Linux NVIDIA GPU machine, run ./setup.sh and ./start.sh there, "
                      "and press Resume — it continues from training. See README: Mac users.")

    def download(self, url, dst):
        self.log(f"Downloading base checkpoint to {dst}")
        tmp = dst.with_suffix(".part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    self.set_progress("Downloading checkpoint", done // (1 << 20), total // (1 << 20), "MB")
                    if self.pause_flag:
                        raise PipelineError("Download interrupted; it will restart on Resume.")
        tmp.rename(dst)

    def st_corpus(self):
        c = self.cfg
        n_test = int(c["test_sentences"])
        tries = 0
        while self.db.count("test") < n_test and tries < 6 and not self.pause_flag:
            tries += 1
            need = n_test - self.db.count("test")
            raw = self.ask_sentences(
                f"Write {need + 8} different sentences in {c['text_language']} to evaluate a finished voice. "
                f"Make them a realistic mix of what this voice will read, including a few harder words, names "
                f"and long sentences, questions and exclamations. {self.domain_line()}\n{RULES}\n"
                'Return {"sentences": ["..."]}')
            good, _ = self.clean_many(raw)
            self.db.insert(good[:need], "test", "openai-test", 1)
        self.log(f"Held-out test set: {self.db.count('test')} sentences.")

        goal = int(self.target_sentences() * float(c["pool_factor"]))
        have = lambda: self.db.q("SELECT COUNT(*) n FROM sent WHERE status!='test'")[0]["n"]
        plan_i = self.db.get("corpus_i", 0)
        calls, max_calls = 0, goal // 15 + 40
        rejects = collections.Counter()
        lock = threading.Lock()

        def job(i):
            topic = TOPICS[i % len(TOPICS)]
            style = STYLES[(i // len(TOPICS)) % len(STYLES)]
            length = LENGTHS[i % len(LENGTHS)]
            raw = self.ask_sentences(
                f"Write 40 different sentences in {c['text_language']}.\nTopic: {topic}.\nStyle: {style}.\n"
                f"Length: {length}.\n{self.domain_line()}\n{RULES}\n"
                'Return {"sentences": ["..."]}')
            good, rej = self.clean_many(raw)
            with lock:
                rejects.update(rej)
            return self.db.insert(good, "pool", "openai", 1)

        while have() < goal and calls < max_calls and not self.pause_flag:
            batch = list(range(plan_i, plan_i + 4))
            plan_i += 4
            calls += 4
            ths, res = [], []
            for i in batch:
                t = threading.Thread(target=lambda i=i: res.append(self._safe(job, i)), daemon=True)
                t.start()
                ths.append(t)
            for t in ths:
                t.join()
            self.db.set("corpus_i", plan_i)
            n = have()
            self.set_progress("Writing sentences", min(n, goal), goal, f"{sum(r or 0 for r in res)} new this batch")
            if all(r is None for r in res):
                raise PipelineError("OpenAI requests keep failing — check the key and model name in Settings.")
        if self.pause_flag:
            return None
        if rejects:
            self.log("Dropped while cleaning: " + ", ".join(f"{k} {v}" for k, v in rejects.most_common()))
        self.log(f"Sentence pool ready: {have():,} candidates.")
        return "select"

    def _safe(self, fn, *a):
        try:
            return fn(*a)
        except Exception as e:
            self.log(f"OpenAI: {e}", "warn")
            return None

    def domain_line(self):
        d = (self.cfg.get("domain") or "").strip()
        return f"About one sentence in four should relate to this use case: {d}." if d else ""

    def st_select(self):
        target = float(self.cfg["target_minutes"]) * 60
        rows = self.db.q("SELECT text, status, dur FROM sent WHERE status IN ('queued','generated','verified','approved')")
        # Calibrate the text-length estimate against real clip durations of this voice.
        done = [r for r in rows if r["status"] != "queued" and r["dur"]]
        est_done = sum(tp.est_sec(r["text"]) for r in done)
        ratio = (sum(r["dur"] for r in done) / est_done) if est_done > 60 else 1.0
        cur = sum(r["dur"] if (r["status"] != "queued" and r["dur"]) else tp.est_sec(r["text"]) * ratio for r in rows)
        need = target - cur
        if need > 30:
            cands = self.db.q("SELECT id, text, type FROM sent WHERE status='pool'")
            chosen = tp.pick_balanced(cands, need / ratio)
            rnd = self.db.get("round", 1)
            self.db.many("UPDATE sent SET status='queued', round=? WHERE id=?", [(rnd, r["id"]) for r in chosen])
            self.log(f"Picked {len(chosen):,} sentences (~{sum(tp.est_sec(r['text']) for r in chosen) * ratio / 60:.0f} min"
                     f"{f', speaking-rate factor {ratio:.2f}' if ratio != 1.0 else ''}) to reach {target / 60:.0f} min.")
            if not chosen:
                self.log("The sentence pool is used up; continuing with what is approved.", "warn")
                self.db.set("topups", 99)
        return "synth"

    def st_synth(self):
        c = self.cfg
        for _pass in range(3):
            rows = self.db.q("SELECT * FROM sent WHERE status='queued' ORDER BY id")
            if not rows:
                return "verify"
            chars = sum(len(r["text"]) for r in rows) * self.credit_factor()
            used = self.db.get("el_chars", 0) * self.credit_factor()
            cap = float(c["max_el_credits"] or 0)
            if cap and used + chars > cap:
                raise PipelineError(f"Credit cap reached: ~{chars:,.0f} more credits needed, cap is {cap:,.0f}. "
                                    "Raise the cap in Settings or lower the target minutes.")
            try:
                sub = self.el.subscription()
                left = int(sub.get("character_limit", 0)) - int(sub.get("character_count", 0))
                self.db.set("el_left", left)
                if left < chars:
                    raise PipelineError(f"Not enough ElevenLabs credits: need ~{chars:,.0f}, {left:,} left.")
            except ApiError:
                pass
            self.log(f"Generating {len(rows):,} clips (~{chars:,.0f} credits).")
            self.run_parallel(rows, self.synth_one, "Generating speech", c["concurrency"])
            if self.pause_flag:
                return None
        left = self.db.q("SELECT id FROM sent WHERE status='queued'")
        if left:
            self.db.many("UPDATE sent SET status='rejected', note='generation failed' WHERE id=?",
                         [(r["id"],) for r in left])
            self.log(f"{len(left)} sentences failed to generate three times and were dropped.", "warn")
        return "verify"

    def synth_one(self, r):
        seed = (int(self.cfg["seed"]) + int(r["attempts"]) * 7919) % 4294967295
        f = self.tts(r["text"], seed, expressive=(r["source"] == "openai-expressive"))
        sr = self.sr()
        name = f"{utt(r)}_a{r['attempts']}.wav"
        aq.write_wav(RAW / name, f, sr)
        q = aq.analyze(f, sr, r["text"], self.qc_cfg())
        self.db.update(r["id"], status="generated", qc=q, dur=q["out_dur"], raw=name, wer=None, asr=None,
                       errwords=None)

    def st_verify(self):
        for _pass in range(2):
            rows = self.db.q("SELECT * FROM sent WHERE status='generated' ORDER BY id")
            if not rows:
                break
            self.run_parallel(rows, self.verify_one, "Transcribing clips", min(6, int(self.cfg["concurrency"]) + 1))
            if self.pause_flag:
                return None
        rest = self.db.q("SELECT id FROM sent WHERE status='generated'")
        if rest:
            self.log(f"{len(rest)} clips could not be transcribed; judged on audio checks only.", "warn")
            self.db.many("UPDATE sent SET status='verified' WHERE id=?", [(r["id"],) for r in rest])
        self.recompute_rates()
        return "repair"

    def verify_one(self, r):
        f, sr = aq.read_wav(RAW / r["raw"])
        q = json.loads(r["qc"])
        text = self.stt(aq.clean(f, q, sr), sr)
        w, bad = tp.align(r["text"], text, self.english)
        self.db.update(r["id"], status="verified", wer=w, asr=text, errwords=bad)

    def recompute_rates(self):
        rows = self.db.q("SELECT id, qc FROM sent WHERE status IN ('verified','approved') AND qc IS NOT NULL")
        qs = [(r["id"], json.loads(r["qc"])) for r in rows]
        rates = [q["rate"] for _, q in qs if q.get("rate", 0) > 0]
        if len(rates) < 8:
            return
        med = statistics.median(rates)
        mad = statistics.median([abs(x - med) for x in rates]) or 0.01
        self.db.set("rate_med", med)
        upd = []
        for sid, q in qs:
            if q.get("rate", 0) > 0:
                q["z"] = (q["rate"] - med) / (1.4826 * mad)
                upd.append((json.dumps(q), sid))
        self.db.many("UPDATE sent SET qc=? WHERE id=?", upd)

    def st_repair(self):
        c = self.cfg
        rows = self.db.q("SELECT * FROM sent WHERE status='verified'")
        approve, requeue, reject, why = [], [], [], collections.Counter()
        for r in rows:
            if r["decision"] == "approve":
                approve.append(r)
                continue
            if r["decision"] == "reject":
                reject.append(r)
                continue
            p = self.problems(r)
            why.update(p)
            if not p:
                approve.append(r)
            elif r["attempts"] + 1 < int(c["max_attempts"]):
                requeue.append(r)
            else:
                reject.append(r)
        self.db.many("UPDATE sent SET status='approved' WHERE id=?", [(r["id"],) for r in approve])
        self.db.many("UPDATE sent SET status='rejected', note='failed checks' WHERE id=?", [(r["id"],) for r in reject])
        self.db.many("UPDATE sent SET status='queued', attempts=attempts+1 WHERE id=?", [(r["id"],) for r in requeue])
        if rows:
            self.log(f"Checked {len(rows):,}: {len(approve):,} approved, {len(requeue):,} to retry with a new seed, "
                     f"{len(reject):,} rejected."
                     + (" Issues: " + ", ".join(f"{aq.FLAG_TEXT.get(k, k)} {v}" for k, v in why.most_common())
                        if why else ""))
        if requeue:
            return "synth"
        appr = self.db.q("SELECT COALESCE(SUM(dur),0) s FROM sent WHERE status='approved'")[0]["s"]
        target = float(c["target_minutes"]) * 60
        topups = self.db.get("topups", 0)
        if appr < target * 0.92 and topups < 4 and self.db.count("pool") > 0:
            self.db.set("topups", topups + 1)
            self.log(f"Approved audio is {appr / 60:.1f} of {target / 60:.0f} min — topping up.")
            return "select"
        return "export"

    def st_export(self):
        rows = self.db.q("SELECT * FROM sent WHERE status='approved' ORDER BY id")
        if len(rows) < 100:
            raise PipelineError(f"Only {len(rows)} approved clips — too few to train. Raise target minutes "
                                "or loosen the quality thresholds in Settings, then jump back to Pick script.")
        wanted, m1, m2, sec = set(), [], [], 0.0
        for k, r in enumerate(rows):
            name = f"{utt(r)}.wav"
            wanted.add(name)
            dst = DSWAV / name
            q = json.loads(r["qc"])
            if r["exported"] != r["raw"] or not dst.exists():
                f, sr = aq.read_wav(RAW / r["raw"])
                aq.write_wav(dst, aq.clean(f, q, sr), sr)
                self.db.update(r["id"], exported=r["raw"])
            sec += q["out_dur"]
            t = _re.sub(r"[|\r\n]+", " ", r["text"]).strip()
            m1.append(f"{utt(r)}|{t}")
            m2.append(f"{name}|{t}")
            if k % 50 == 0:
                self.set_progress("Building dataset", k, len(rows))
        for p in DSWAV.glob("*.wav"):
            if p.name not in wanted:
                p.unlink()
        (DS / "metadata.csv").write_text("\n".join(m1) + "\n", encoding="utf-8")
        (DS / "metadata_piper1.csv").write_text("\n".join(m2) + "\n", encoding="utf-8")
        tests = self.db.q("SELECT text FROM sent WHERE status='test' ORDER BY id")
        (DS / "test_sentences.txt").write_text("\n".join(t["text"] for t in tests) + "\n", encoding="utf-8")
        self.log(f"Dataset: {len(rows):,} clips, {sec / 60:.1f} min, in {DS}")
        return "train"

    def batch_size(self):
        b = int(self.cfg["batch_size"] or 0)
        if b:
            return b
        gb = (self.db.get("vram_mb", 0) or 0) / 1024
        return 8 if gb < 10 else 12 if gb < 14 else 16 if gb < 20 else 32 if gb < 30 else 48

    def st_train(self):
        c = self.cfg
        dev = self.probe_device()
        if dev["device"] == "none":
            self.handoff()
        if not BASE_CKPT.exists():
            self.download(c["base_ckpt_url"], BASE_CKPT)
        rnd = self.db.get("round", 1)
        key = f"train_r{rnd}"
        info = self.db.get(key) or {}
        if info.get("done"):
            return "evaluate"
        # Old Piper checkpoints (e.g. Lessac) can't be *resumed* by the current trainer: newer PyTorch refuses
        # to unpickle them and their hyperparameters don't match. So the first run *warm-starts* from them
        # (copies every compatible weight, fresh optimizer, epoch counter starts at 0). Later runs resume
        # our own, fully compatible checkpoints.
        resume = self.pick_resume() if (rnd > 1 or info.get("started")) else None
        self.prune_runs(resume, rnd)
        free_gb = shutil.disk_usage(WORK).free / 2**30
        if free_gb < 4:
            raise PipelineError(f"Only {free_gb:.1f} GB of disk space is free. Training needs about 4 GB for "
                                "checkpoints. Free up space (README: Disk space), then press Resume.")
        if rnd == 1:
            max_ep = int(c["epochs_first"])
        else:
            prev = self.db.get(f"train_r{rnd - 1}") or {}
            max_ep = info.get("max_epochs") or int(prev.get("max_epochs", c["epochs_first"])) + int(c["epochs_improve"])
        if resume and info.get("last_epoch") is not None and max_ep <= int(info["last_epoch"]) + 1:
            raise PipelineError(f"This round is already at epoch {info['last_epoch']}; the target of {max_ep} epochs "
                                "is not higher. Raise the epochs in Settings → Training, then Restart from Train.")
        info.update(started=True, max_epochs=max_ep)
        self.db.set(key, info)
        voice = c["voice_name"]
        cmd = [str(PIPER_PY), str(ROOT / "train_launcher.py"), "fit",
               "--data.voice_name", voice,
               "--data.csv_path", str(DS / "metadata_piper1.csv"),
               "--data.audio_dir", str(DSWAV) + "/",
               "--model.sample_rate", str(self.sr()),
               "--data.espeak_voice", c["espeak_voice"],
               "--data.cache_dir", str(WORK / f"cache_r{rnd}") + "/",
               "--data.config_path", str(WORK / f"{voice}.onnx.json"),
               "--data.batch_size", str(self.batch_size()),
               "--trainer.max_epochs", str(max_ep),
               "--trainer.default_root_dir", str(RUNS),
               "--data.validation_split", "0.05"]
        if resume:
            cmd += ["--ckpt_path", str(resume), "--model.warmstart_ckpt", "null"]
        else:
            cmd += ["--model.warmstart_ckpt", str(BASE_CKPT)]
        if dev["device"] in ("mps", "cpu"):
            cmd += ["--trainer.accelerator", dev["device"], "--trainer.devices", "1"]
        if dev["device"] == "mps":
            # Apple GPU (macOS 14) can't run conv1d on sequences > 65,536 samples. Training uses short
            # segments (fine), but the end-of-epoch preview synthesizes whole sentences and crashes.
            # Skip that preview + its MOS score; Voice Forge's own evaluation stage replaces it.
            cmd += ["--data.num_test_examples", "0", "--model.mos_metric", "none"]
        cmd += shlex.split(c.get("extra_train_args") or "")
        prev_max = (self.db.get(f"train_r{rnd - 1}") or {}).get("max_epochs", 0) if rnd > 1 else 0
        start_ep = int(info.get("last_epoch", prev_max)) if resume else 0
        self.train = {"round": rnd, "epoch": start_ep, "start": start_ep, "max": max_ep, "pct": 0, "losses": {},
                      "t0": time.time(), "ep_times": []}
        self.log(f"Training round {rnd}: epochs {start_ep} → {max_ep}, batch size {self.batch_size()}, "
                 + (f"resuming {Path(resume).name}." if resume else f"warm-starting from {BASE_CKPT.name}."))
        self.log("$ " + " ".join(shlex.quote(x) for x in cmd))
        tlog = open(WORK / f"train_r{rnd}.log", "a", encoding="utf-8")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
        if dev["device"] == "mps":
            self.log("Apple GPU training is experimental and much slower than an NVIDIA card. "
                     "Good for a short test; use a cloud GPU for the full run.", "warn")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(PIPER_DIR),
                                     env=env, start_new_session=True)
        ep_re = _re.compile(r"Epoch (\d+):\s*(\d+)%")
        loss_re = _re.compile(r"(loss[\w]*)=([\d.]+)")
        tail = collections.deque(maxlen=40)
        buf, last_ep = b"", None
        fd = self.proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            buf += chunk
            parts = _re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                line = p.decode("utf-8", "replace").strip()
                if not line:
                    continue
                m = ep_re.search(line)
                if m:
                    ep, pct = int(m.group(1)), int(m.group(2))
                    if ep != last_ep:
                        now = time.time()
                        if last_ep is not None:
                            self.train["ep_times"] = (self.train["ep_times"] + [now - self.train["ep_t"]])[-20:]
                            tlog.write(line + "\n")
                        self.train["ep_t"] = now
                        last_ep = ep
                    self.train.update(epoch=ep, pct=pct)
                    if ep != info.get("last_epoch"):
                        info["last_epoch"] = ep
                        self.db.set(key, info)
                    for k, v in loss_re.findall(line):
                        self.train["losses"][k] = float(v)
                    done = ep - start_ep + pct / 100
                    tot = max(1, max_ep - start_ep)
                    et = self.train["ep_times"]
                    eta = (statistics.mean(et) * (tot - done)) if et else None
                    self.progress = {"label": f"Training round {rnd}", "done": round(done, 2), "total": tot,
                                     "detail": f"epoch {ep} of {max_ep}", "eta": eta}
                else:
                    tail.append(line)
                    tlog.write(line + "\n")
                    if not _re.search(r"Warning|warn\(|^\s*$", line):
                        self.log("train: " + line[:300])
        rc = self.proc.wait()
        tlog.close()
        self.proc = None
        if self.pause_flag:
            self.log("Training paused; it will resume from the latest checkpoint.")
            return None
        if rc != 0:
            raise PipelineError(f"Training exited with code {rc}. Last output:\n" + "\n".join(list(tail)[-15:]))
        info["done"] = True
        self.db.set(key, info)
        return "evaluate"

    def st_evaluate(self):
        c = self.cfg
        rnd = self.db.get("round", 1)
        voice = c["voice_name"]
        ckpt = self.latest_ckpt()
        if not ckpt:
            raise PipelineError("No trained checkpoint found.")
        outdir = OUT / f"round_{rnd}"
        outdir.mkdir(parents=True, exist_ok=True)
        onnx = outdir / f"{voice}.onnx"
        if not onnx.exists():
            self.log(f"Exporting ONNX from {ckpt.name}")
            self.set_progress("Exporting ONNX", 0, 1)
            try:
                self.piper_run([ROOT / "export_launcher.py", "--checkpoint", ckpt, "--output-file", onnx],
                               timeout=1800)
            except PipelineError:
                onnx.unlink(missing_ok=True)  # never keep a half-written model
                raise
        shutil.copy(WORK / f"{voice}.onnx.json", outdir / f"{voice}.onnx.json")
        tests = self.db.q("SELECT id, text FROM sent WHERE status='test' ORDER BY id")
        # 1) Piper reads the held-out test sentences (offline, always).
        edir = EVAL / f"r{rnd}"
        edir.mkdir(parents=True, exist_ok=True)
        jobs = [{"text": t["text"], "out": str(edir / f"test_{t['id']}.wav")} for t in tests]
        (edir / "jobs.json").write_text(json.dumps(jobs))
        self.set_progress("Your Piper voice reads the test sentences", 0, 1)
        self.piper_run([ROOT / "piper_synth.py", onnx, edir / "jobs.json"], timeout=3600)
        latest = OUT / "latest"
        latest.mkdir(exist_ok=True)
        for p in outdir.glob(f"{voice}.onnx*"):
            shutil.copy(p, latest / p.name)
        # 2) Optional: ElevenLabs reference clips for side-by-side listening.
        refs_ok = bool(c.get("eval_with_apis", True))
        for k, t in enumerate(tests if refs_ok else []):
            ref = REFS / f"test_{t['id']}.wav"
            if ref.exists():
                continue
            try:
                aq.write_wav(ref, self.tts(t["text"], int(c["seed"])), self.sr())
            except ApiError as e:
                self.log(f"Skipping ElevenLabs reference clips ({e}). Your Piper test clips are still available.", "warn")
                refs_ok = False
                break
            self.set_progress("ElevenLabs reference clips", k + 1, len(tests))
        # 3) Optional: transcribe to score word errors.
        items, piper_bad, ref_bad, scored = [], collections.Counter(), set(), bool(c.get("eval_with_apis", True))
        for k, t in enumerate(tests):
            it = {"id": t["id"], "text": t["text"], "wer": None, "asr": None, "ref_wer": None, "ref_asr": None,
                  "errs": [], "has_ref": (REFS / f"test_{t['id']}.wav").exists()}
            if scored:
                try:
                    pf, psr = aq.read_wav(edir / f"test_{t['id']}.wav")
                    it["asr"] = self.stt(pf, psr)
                    it["wer"], it["errs"] = tp.align(t["text"], it["asr"], self.english)
                    piper_bad.update(w for w in it["errs"] if len(w) >= 3)
                    if it["has_ref"]:
                        rf, rsr = aq.read_wav(REFS / f"test_{t['id']}.wav")
                        it["ref_asr"] = self.stt(rf, rsr)
                        it["ref_wer"], rbad = tp.align(t["text"], it["ref_asr"], self.english)
                        ref_bad.update(rbad)
                except ApiError as e:
                    self.log(f"Skipping automatic scoring ({e}). Listen to the test clips in the dashboard instead.",
                             "warn")
                    scored = False
            items.append(it)
            self.set_progress("Scoring the new voice", k + 1, len(tests))
        problem = [w for w, _ in piper_bad.most_common() if w not in ref_bad] if scored else []
        wers = [i["wer"] for i in items if i["wer"] is not None]
        rwers = [i["ref_wer"] for i in items if i["ref_wer"] is not None]
        approved = self.db.q("SELECT COUNT(*) n, COALESCE(SUM(dur),0) s FROM sent WHERE status='approved'")[0]
        rep = {"round": rnd, "at": time.time(), "wer": statistics.mean(wers) if scored and wers else None,
               "ref_wer": statistics.mean(rwers) if rwers else None, "items": items,
               "problem_words": problem[:60], "clips": approved["n"], "minutes": approved["s"] / 60,
               "onnx": str(onnx.relative_to(ROOT)), "epochs": (self.db.get(f"train_r{rnd}") or {}).get("max_epochs")}
        self.db.set(f"report_r{rnd}", rep)
        if rep["wer"] is not None:
            self.log(f"Round {rnd}: word error {rep['wer'] * 100:.1f}%"
                     + (f" (ElevenLabs original {rep['ref_wer'] * 100:.1f}%)" if rep["ref_wer"] is not None else "")
                     + f". Problem words: {', '.join(problem[:12]) or 'none'}")
        else:
            self.log(f"Round {rnd}: voice exported and test sentences rendered (not scored). Listen in the dashboard.")
        return "improve"

    def st_improve(self):
        c = self.cfg
        rnd = self.db.get("round", 1)
        rep = self.db.get(f"report_r{rnd}") or {}
        scored = rep.get("wer") is not None
        good_enough = scored and rep["wer"] <= max(float(c["target_wer"]), (rep.get("ref_wer") or 0) + 0.01)
        words = rep.get("problem_words", [])[: int(c["improve_words"])]
        if rnd >= int(c["max_rounds"]) or good_enough or not words:
            final = OUT / "final"
            final.mkdir(exist_ok=True)
            for p in (OUT / "latest").glob("*"):
                shutil.copy(p, final / p.name)
            why = ("reached the round limit" if rnd >= int(c["max_rounds"]) else
                   "hit the quality target" if good_enough else
                   "no scores to improve from (scoring skipped)" if not scored else "no specific problem words left")
            self.log(f"Done after {rnd} round(s): {why}. Final voice in {final}")
            return "done"
        per = int(c["improve_per_word"])
        added = 0
        pool = self.db.q("SELECT id, text FROM sent WHERE status='pool'")
        take = []
        for w in words:
            rx = _re.compile(rf"(?<!\w){_re.escape(w)}(?!\w)", _re.I)
            hits = [p for p in pool if rx.search(p["text"])][: max(1, per // 2)]
            take += hits
        self.db.many("UPDATE sent SET status='queued', round=? WHERE id=?", [(rnd + 1, p["id"]) for p in take])
        added += len(take)
        for i in range(0, len(words), 10):
            chunk = words[i:i + 10]
            raw = self._safe(self.ask_sentences,
                             f"The voice mispronounces these words: {', '.join(chunk)}.\n"
                             f"Write {per} different sentences in {c['text_language']} for EACH word, each "
                             f"containing that exact word, in varied positions and contexts.\n{RULES}\n"
                             'Return {"sentences": ["..."]}') or []
            good, _ = self.clean_many(raw)
            added += self.db.insert(good, "queued", "openai-improve", rnd + 1)
        self.db.set("round", rnd + 1)
        self.db.set("topups", 0)
        self.log(f"Round {rnd + 1}: added {added} targeted sentences for {len(words)} problem words.")
        return "synth"

    def st_expressive(self):
        """Write emotional sentences, record them with a livelier ElevenLabs setting, then fine-tune the
        existing voice on old + new clips as a new round. Nothing already recorded is regenerated."""
        c = self.cfg
        target = float(c["expressive_minutes"]) * 60
        have = lambda: sum(tp.est_sec(r["text"]) for r in self.db.q(
            "SELECT text FROM sent WHERE source='openai-expressive' AND status IN "
            "('queued','generated','verified','approved')"))
        i = self.db.get("expr_i", 0)
        calls = 0
        while have() < target and calls < 80 and not self.pause_flag:
            emo, form = EMOTIONS[i % len(EMOTIONS)], EXPR_FORMS[(i // len(EMOTIONS)) % len(EXPR_FORMS)]
            raw = self._safe(self.ask_sentences,
                             f"Write 25 different {form} in {c['text_language']} that sound clearly {emo}.\n"
                             "The emotion must come through in the words and punctuation alone (no stage "
                             "directions, no descriptions like 'she laughed', no brackets). Use exclamation "
                             "marks, questions, commas and short pauses naturally. Mix lengths from four to "
                             f"twenty-five words. {self.domain_line()}\n{RULES}\n"
                             'Return {"sentences": ["..."]}') or []
            good, _ = self.clean_many(raw)
            rnd_next = self.db.get("round", 1) + 1
            self.db.insert(good, "queued", "openai-expressive", rnd_next)
            i += 1
            calls += 1
            self.db.set("expr_i", i)
            self.set_progress("Writing expressive sentences", min(have(), target) / 60, target / 60, "minutes")
        if self.pause_flag:
            return None
        rnd = self.db.get("round", 1)
        nxt = rnd + 1
        prev = self.db.get(f"train_r{rnd}") or {}
        self.db.set(f"train_r{nxt}", {"max_epochs": int(prev.get("max_epochs") or 0) + int(c["epochs_improve"])})
        self.db.set(f"report_r{nxt}", None)
        for p in (OUT / f"round_{nxt}").glob("*.onnx*"):
            p.unlink()
        self.db.set("round", nxt)
        self.db.set("topups", 99)
        n = self.db.q("SELECT COUNT(*) n FROM sent WHERE source='openai-expressive' AND status='queued'")[0]["n"]
        self.log(f"Expressive round {rnd + 1}: {n} emotional sentences queued (stability "
                 f"{c['expressive_stability']}, style {c['expressive_style']}). Existing clips are reused.")
        return "synth"

    # ---------- dashboard data ----------
    def snapshot(self, since=0):
        rows = self.db.q("SELECT status, COUNT(*) n, COALESCE(SUM(dur),0) d FROM sent GROUP BY status")
        counts = {r["status"]: r["n"] for r in rows}
        appr = next((r["d"] for r in rows if r["status"] == "approved"), 0)
        stage = self.db.get("stage", "check")
        order = [k for k, _ in STAGES]
        si = len(order) if stage == "done" else order.index(stage)
        rnd = self.db.get("round", 1)
        reports = [r for r in (self.db.get(f"report_r{i}") for i in range(1, rnd + 1)) if r]
        with self.log_lock:
            logs = [e for e in self.logs if e["id"] > since][-400:]
        cf = self.credit_factor()
        return {
            "status": self.status, "message": self.message, "stage": stage, "round": rnd,
            "stages": [{"key": k, "label": l, "state": "done" if i < si else "active" if i == si else "todo"}
                       for i, (k, l) in enumerate(STAGES)],
            "progress": {k: v for k, v in self.progress.items() if k != "t0"},
            "counts": counts, "approved_sec": appr, "target_sec": float(self.cfg["target_minutes"]) * 60,
            "el_used": round((self.db.get("el_chars", 0) or 0) * cf), "el_left": self.db.get("el_left"),
            "oa_tokens": self.db.get("oa_tokens", 0), "train": {k: v for k, v in self.train.items()
                                                               if k not in ("ep_t", "ep_times", "t0")},
            "reports": reports, "logs": logs, "final": (OUT / "final" / f"{self.cfg['voice_name']}.onnx").exists(),
            "configured": bool(self.cfg["elevenlabs_api_key"] and self.cfg["openai_api_key"] and self.cfg["voice_id"]),
        }
