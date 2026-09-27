"""Voice Forge dashboard server. Run ./start.sh, then open http://127.0.0.1:8765"""
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

import audioqc as aq
from apis import ApiError, ElevenLabs, OpenAI
from pipeline import (DEFAULT_CONFIG, EVAL, OUT, PIPER_DIR, PIPER_PY, RAW, REFS, ROOT, WORK, PipelineError,
                      Runner, load_config, save_config, utt)

app = FastAPI(title="Voice Forge")
runner = Runner()
SECRET = ("elevenlabs_api_key", "openai_api_key")


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "dashboard.html").read_text(encoding="utf-8")


@app.get("/api/config")
def get_config():
    cfg = load_config()
    out = {k: v for k, v in cfg.items() if k not in SECRET}
    for k in SECRET:
        out[k] = ""
        out["_" + k] = ("…" + cfg[k][-4:]) if cfg[k] else ""
    return out


@app.post("/api/config")
def set_config(data: dict = Body(...)):
    cfg = load_config()
    for k, default in DEFAULT_CONFIG.items():
        if k not in data:
            continue
        v = data[k]
        if k in SECRET and not v:
            continue
        try:
            if isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, int):
                v = int(float(v))
            elif isinstance(default, float):
                v = float(v)
            else:
                v = str(v).strip()
        except (TypeError, ValueError):
            raise HTTPException(400, f"{k}: not a valid value")
        cfg[k] = v
    save_config(cfg)
    if not runner.running:
        runner.cfg = cfg
    return {"ok": True, "note": "Saved. Changes apply from the next stage." if runner.running else "Saved."}


@app.get("/api/voices")
def voices():
    cfg = load_config()
    if not cfg["elevenlabs_api_key"]:
        raise HTTPException(400, "Save your ElevenLabs key first.")
    try:
        vs = ElevenLabs(cfg["elevenlabs_api_key"]).voices()
    except ApiError as e:
        raise HTTPException(400, str(e))
    return [{"id": v["voice_id"], "name": v["name"], "category": v.get("category", "")} for v in vs]


@app.post("/api/check-keys")
def check_keys():
    cfg = load_config()
    out = {}
    if cfg["elevenlabs_api_key"]:
        try:
            sub = ElevenLabs(cfg["elevenlabs_api_key"]).subscription()
            left = int(sub.get("character_limit", 0)) - int(sub.get("character_count", 0))
            out["elevenlabs"] = f"OK: plan {sub.get('tier')}, {left:,} credits left"
            runner.db.set("el_left", left)
        except Exception as e:
            out["elevenlabs"] = f"Not working: {e}"
    else:
        out["elevenlabs"] = "Not set"
    if cfg["openai_api_key"]:
        try:
            OpenAI(cfg["openai_api_key"]).check_model(cfg["openai_model"])
            out["openai"] = f"OK: {cfg['openai_model']} available"
        except Exception as e:
            out["openai"] = f"Not working: {e}"
    else:
        out["openai"] = "Not set"
    return out


def find_voice():
    """Best available exported voice: final, then latest, then newest round."""
    rounds = sorted(OUT.glob("round_*"), key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
                    reverse=True)
    for d in [OUT / "final", OUT / "latest", *rounds]:
        for onnx in sorted(d.glob("*.onnx")):
            if onnx.with_suffix(".onnx.json").exists():
                return onnx
    return None


INFER_KEYS = (("noise_scale", "noise_scale", 0.667), ("noise_w_scale", "noise_w", 0.8), ("length_scale", "length_scale", 1.0))


@app.get("/api/voice")
def voice_info():
    v = find_voice()
    out = {"voice": str(v.relative_to(ROOT)) if v else None, "ready": bool(v and PIPER_PY.exists())}
    if v:
        inf = json.loads(v.with_suffix(".onnx.json").read_text()).get("inference", {})
        out["defaults"] = {k: inf.get(jk, d) for k, jk, d in INFER_KEYS}
    return out


@app.post("/api/voice-defaults")
def voice_defaults(data: dict = Body(...)):
    """Store the slider values in the voice's .onnx.json so every Piper app uses them."""
    v = find_voice()
    if not v:
        raise HTTPException(404, "No exported voice yet.")
    changed = []
    for folder in {v.parent, OUT / "final", OUT / "latest"}:
        cfgp = folder / v.with_suffix(".onnx.json").name
        if not cfgp.exists():
            continue
        cfg = json.loads(cfgp.read_text())
        inf = cfg.setdefault("inference", {})
        for k, jk, _ in INFER_KEYS:
            if data.get(k) not in (None, ""):
                inf[jk] = round(float(data[k]), 3)
        cfgp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        changed.append(str(cfgp.relative_to(ROOT)))
    return {"ok": True, "files": changed}


@app.post("/api/expressive")
def expressive():
    if runner.running:
        raise HTTPException(400, "Pause the pipeline first.")
    runner.goto("expressive")
    runner.start()
    return {"ok": True}


@app.post("/api/speak")
def speak(data: dict = Body(...)):
    text = " ".join(str(data.get("text", "")).split())[:1000]
    if not text:
        raise HTTPException(400, "Type something to say.")
    v = find_voice()
    if not v:
        raise HTTPException(404, "No exported voice yet.")
    if not PIPER_PY.exists():
        raise HTTPException(400, "Piper isn't installed on this machine (run the setup script with the trainer).")
    out = WORK / "speak.wav"
    jobs = WORK / "speak.json"
    cfg = {k: data.get(k) for k, _, _ in INFER_KEYS if data.get(k) not in (None, "")}
    jobs.write_text(json.dumps([{"text": text, "out": str(out), "cfg": cfg}]))
    r = subprocess.run([str(PIPER_PY), str(ROOT / "piper_synth.py"), str(v), str(jobs)], capture_output=True,
                       text=True, timeout=300, cwd=str(PIPER_DIR))
    if r.returncode or not out.exists():
        raise HTTPException(500, (r.stderr or r.stdout)[-800:])
    return Response(out.read_bytes(), media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/start")
def start():
    runner.start()
    return {"ok": True}


@app.post("/api/pause")
def pause():
    runner.pause()
    return {"ok": True}


@app.post("/api/goto")
def goto(data: dict = Body(...)):
    try:
        runner.goto(data.get("stage", ""))
    except PipelineError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/status")
def status(since: int = 0):
    return runner.snapshot(since)


@app.get("/api/clips")
def clips(filter: str = "recent", limit: int = 60):
    where = {"recent": "status IN ('approved','rejected','verified','generated')",
             "approved": "status='approved'", "rejected": "status='rejected'",
             "queued": "status='queued'", "test": "status='test'"}.get(filter, "1=1")
    order = "updated DESC" if filter in ("recent", "approved", "rejected") else "id"
    rows = runner.db.q(f"SELECT * FROM sent WHERE {where} ORDER BY {order} LIMIT ?", (min(limit, 300),))
    out = []
    for r in rows:
        q = json.loads(r["qc"]) if r["qc"] else {}
        out.append({"id": r["id"], "utt": utt(r), "text": r["text"], "status": r["status"],
                    "attempts": r["attempts"], "wer": r["wer"], "asr": r["asr"], "dur": r["dur"],
                    "has_audio": bool(r["raw"]), "problems": runner.problems(r) if r["qc"] else [],
                    "z": q.get("z"), "note": r["note"], "round": r["round"], "decision": r["decision"]})
    return out


@app.post("/api/decide/{sid}")
def decide(sid: int, data: dict = Body(...)):
    d = data.get("decision")
    if d not in ("approve", "reject", None):
        raise HTTPException(400, "bad decision")
    r = runner.db.q("SELECT status FROM sent WHERE id=?", (sid,))
    if not r:
        raise HTTPException(404)
    st = r[0]["status"]
    new = st
    if d == "approve" and st in ("rejected", "verified"):
        new = "approved"
    elif d == "reject" and st in ("approved", "verified"):
        new = "rejected"
    runner.db.update(sid, decision=d, status=new)
    runner.log(f"Manual decision on utt_{sid:05d}: {d or 'cleared'} (applies at the next dataset build)")
    return {"ok": True, "status": new}


def _wav(path: Path):
    if not path.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/audio/clip/{sid}")
def clip_audio(sid: int, clean: int = 1):
    r = runner.db.q("SELECT raw, qc FROM sent WHERE id=?", (sid,))
    if not r or not r[0]["raw"]:
        raise HTTPException(404)
    f, sr = aq.read_wav(RAW / Path(r[0]["raw"]).name)
    if clean and r[0]["qc"]:
        f = aq.clean(f, json.loads(r[0]["qc"]), sr)
    return Response(aq.wav_bytes(f, sr), media_type="audio/wav")


@app.get("/api/audio/ref/{sid}")
def ref_audio(sid: int):
    return _wav(REFS / f"test_{sid}.wav")


@app.get("/api/audio/eval/{rnd}/{sid}")
def eval_audio(rnd: int, sid: int):
    return _wav(EVAL / f"r{rnd}" / f"test_{sid}.wav")


@app.get("/api/download/{which}")
def download(which: str):
    folder = OUT / ("final" if which == "final" else "latest")
    files = list(folder.glob("*.onnx*"))
    if not files:
        raise HTTPException(404, "No exported voice yet.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
    name = f"{load_config()['voice_name']}_{which}.zip"
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"\n  Voice Forge dashboard: http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
