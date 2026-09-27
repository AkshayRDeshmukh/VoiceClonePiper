"""Runs Piper's trainer (python -m piper.train ...) inside the piper1-gpl venv, with two changes:
1. The ModelCheckpoint monitoring "val_mos" is removed. When the MOS predictor is disabled or can't load
   (e.g. on Apple GPUs) it raises MisconfigurationException at epoch end.
2. The "val_mel" checkpoint keeps only the best VF_KEEP_CKPTS files (default 1) instead of 5, so disk use
   stays around 2 checkpoints (best + last.ckpt, roughly 800 MB each).
Usage: python train_launcher.py fit --data.… (same arguments as piper.train)."""
import os

import piper.train.__main__ as piper_main

keep = int(os.environ.get("VF_KEEP_CKPTS", "1"))
piper_main._DEFAULT_CALLBACKS[:] = [
    cb for cb in piper_main._DEFAULT_CALLBACKS if getattr(cb, "monitor", None) != "val_mos"
]
for cb in piper_main._DEFAULT_CALLBACKS:
    if getattr(cb, "monitor", None) == piper_main._MONITOR:
        cb.save_top_k = keep

if __name__ == "__main__":
    piper_main.main()
