#!/usr/bin/env bash
# macOS (Apple Silicon: M1/M2/M3/M4) setup for Voice Forge.
#   ./setup_mac.sh                 dashboard + data pipeline (recommended)
#   ./setup_mac.sh --with-trainer  also install the Piper trainer for experimental Apple-GPU (mps) training
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
case "$ROOT" in
  *" "*) echo "ERROR: the project path contains a space:"; echo "  $ROOT"
         echo "Piper's build tools can't handle that. Move the folder, e.g.: mv \"$ROOT\" ~/voiceforge"; exit 1;;
esac
say(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/4 Developer tools"
xcode-select -p >/dev/null 2>&1 || { echo "Installing Xcode Command Line Tools — rerun this script when it finishes."; xcode-select --install; exit 1; }
command -v brew >/dev/null 2>&1 || { echo "Install Homebrew first: https://brew.sh  then rerun."; exit 1; }

say "2/4 Homebrew packages"
brew install python@3.11 espeak-ng cmake ninja git wget
PY="$(brew --prefix python@3.11)/bin/python3.11"

say "3/4 Dashboard environment (.venv)"
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ "${1:-}" = "--with-trainer" ]; then
  say "4/4 Piper trainer (experimental on Apple Silicon)"
  mkdir -p third_party
  [ -d third_party/piper1-gpl ] || git clone https://github.com/OHF-voice/piper1-gpl.git third_party/piper1-gpl
  cd third_party/piper1-gpl
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip wheel setuptools
  .venv/bin/pip install -e '.[train]'
.venv/bin/pip install scikit-build cmake ninja onnxscript   # needed for the in-place dev build below
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ./build_monotonic_align.sh
  deactivate
  cd "$ROOT"
  # The eSpeak version Piper pins truncates file paths at 179 characters on macOS, so building
  # inside a long path fails ("Bad vowel file"). Build the phonemizer in a short temp folder
  # (same commit) and copy the results into the project.
  PB="/tmp/pb$$"
  rm -rf "$PB"
  git clone -q "$ROOT/third_party/piper1-gpl" "$PB"
  (cd "$PB" && PATH="$ROOT/third_party/piper1-gpl/.venv/bin:$PATH" python3 setup.py build_ext --inplace)
  cp -R "$PB"/src/piper/espeakbridge*.so "$PB/src/piper/espeak-ng-data" "$ROOT/third_party/piper1-gpl/src/piper/"
  rm -rf "$PB"
  third_party/piper1-gpl/.venv/bin/python -c "import torch, piper.train; from piper import espeakbridge; print('torch', torch.__version__, '| Apple GPU (mps):', torch.backends.mps.is_available())"
  mkdir -p checkpoints
  [ -f checkpoints/base-medium.ckpt ] || wget -O checkpoints/base-medium.ckpt \
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
  echo "Set Settings > Training > Training device to 'mps' to train on the Apple GPU."
else
  say "4/4 Skipping the trainer (training will happen on a GPU machine)"
fi

chmod +x start.sh
say "Setup complete. Run ./start.sh and open http://127.0.0.1:8765"
