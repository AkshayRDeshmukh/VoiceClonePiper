"""Run inside the piper1-gpl venv: synthesize sentences with a Piper ONNX voice.
Usage: python piper_synth.py model.onnx jobs.json
jobs = [{"text": ..., "out": ..., "cfg": {"length_scale": .., "noise_scale": .., "noise_w_scale": ..}}]"""
import json
import subprocess
import sys
import wave

model, jobs_path = sys.argv[1], sys.argv[2]
jobs = json.load(open(jobs_path, encoding="utf-8"))
try:
    from piper import PiperVoice
    try:
        from piper import SynthesisConfig
    except ImportError:
        SynthesisConfig = None
    voice = PiperVoice.load(model)
    for j in jobs:
        cfg = {k: float(v) for k, v in (j.get("cfg") or {}).items() if v not in (None, "")}
        with wave.open(j["out"], "wb") as wf:
            if hasattr(voice, "synthesize_wav"):
                syn = SynthesisConfig(**cfg) if (SynthesisConfig and cfg) else None
                voice.synthesize_wav(j["text"], wf, syn_config=syn)
            else:
                voice.synthesize(j["text"], wf)
        print("ok", j["out"], flush=True)
except ImportError:
    for j in jobs:
        extra = []
        for k, flag in (("length_scale", "--length-scale"), ("noise_scale", "--noise-scale"),
                        ("noise_w_scale", "--noise-w-scale")):
            if (j.get("cfg") or {}).get(k) not in (None, ""):
                extra += [flag, str(j["cfg"][k])]
        subprocess.run([sys.executable, "-m", "piper", "-m", model, "-f", j["out"], *extra, "--", j["text"]], check=True)
        print("ok", j["out"], flush=True)
