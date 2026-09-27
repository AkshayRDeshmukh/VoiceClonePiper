#!/usr/bin/env bash
# One-time setup for Voice Forge.
# Tested target: Ubuntu 22.04 / 24.04 (native or WSL2) with an NVIDIA GPU.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
case "$ROOT" in
  *" "*) echo "ERROR: the project path contains a space:"; echo "  $ROOT"
         echo "Piper's build tools can't handle that. Move the folder, e.g.: mv \"$ROOT\" ~/voiceforge"; exit 1;;
esac
say(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"   # cloud GPU containers often run as root without sudo
say "1/6 System packages"
$SUDO apt-get update
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y python3 python3-venv python3-dev python3-pip build-essential cmake ninja-build \
  git wget curl espeak-ng

say "2/6 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "WARNING: nvidia-smi not found. Install the NVIDIA driver first."
  echo "         On WSL2: install the normal Windows NVIDIA driver; do NOT install a Linux driver inside WSL."
fi

say "3/6 Dashboard environment (.venv)"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

say "4/6 Piper trainer (third_party/piper1-gpl)"
mkdir -p third_party
[ -d third_party/piper1-gpl ] || git clone https://github.com/OHF-voice/piper1-gpl.git third_party/piper1-gpl
cd third_party/piper1-gpl
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install -e '.[train]'
.venv/bin/pip install scikit-build cmake ninja onnxscript   # needed for the in-place dev build below
# shellcheck disable=SC1091
source .venv/bin/activate
./build_monotonic_align.sh
python3 setup.py build_ext --inplace
deactivate
cd "$ROOT"

say "5/6 Verify trainer + CUDA"
third_party/piper1-gpl/.venv/bin/python - <<'PY'
import torch, piper.train
from piper import espeakbridge  # phonemizer built OK
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA available:", ok)
if ok:
    print("GPU:", torch.cuda.get_device_name(0), "|", torch.cuda.get_device_properties(0).total_memory // 2**30, "GB")
else:
    print("!! PyTorch cannot see a GPU. See README > Troubleshooting before training.")
PY

say "6/6 Base checkpoint (Lessac medium, ~800 MB)"
mkdir -p checkpoints
[ -f checkpoints/base-medium.ckpt ] || wget -O checkpoints/base-medium.ckpt \
  "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"

chmod +x start.sh
say "Setup complete. Run ./start.sh and open http://127.0.0.1:8765"
