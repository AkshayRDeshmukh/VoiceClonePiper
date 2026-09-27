"""Runs Piper's ONNX export (python -m piper.train.export_onnx ...) inside the piper1-gpl venv.
Newer PyTorch switched torch.onnx.export's default to the dynamo exporter (needs onnxscript and doesn't
match what Piper was built for). Prefer the classic exporter; fall back to the new one if it's unavailable.
Usage: python export_launcher.py --checkpoint X.ckpt --output-file voice.onnx"""
import inspect
import runpy

import torch

_orig_export = torch.onnx.export


def _export(*args, **kwargs):
    if "dynamo" not in kwargs and "dynamo" in inspect.signature(_orig_export).parameters:
        try:
            return _orig_export(*args, dynamo=False, **kwargs)
        except Exception as exc:  # classic exporter removed or failed: try the new one
            print(f"[voiceforge] classic ONNX exporter failed ({exc}); trying the new exporter", flush=True)
    return _orig_export(*args, **kwargs)


torch.onnx.export = _export

if __name__ == "__main__":
    runpy.run_module("piper.train.export_onnx", run_name="__main__", alter_sys=True)
