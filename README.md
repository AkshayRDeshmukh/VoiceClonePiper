# Voice Clone Piper

Turns an ElevenLabs voice into a local Piper voice (`.onnx` + `.onnx.json`) automatically.
You set keys and a voice once, press **Start**, and watch the dashboard.

```
OpenAI writes sentences → clean + pick a balanced script → ElevenLabs speaks them
→ audio checks + transcript check → retry bad takes with a new seed → build dataset
→ fine-tune Piper on your GPU → export ONNX → test on held-out sentences
→ write new sentences for mispronounced words → train again → final ONNX
```

Everything is resumable: pause, close the terminal, reboot — press **Resume** and it continues from the same stage.

## What you need

| Thing | Requirement |
|---|---|
| For sentences, audio and dataset (stages 1–7) | **Any Mac (incl. M1)**, Linux or WSL2. No GPU needed |
| For training (stages 8–10) | Linux with an **NVIDIA GPU**, 8 GB VRAM minimum and 16–24 GB comfortable. Your own PC or a rented cloud GPU |
| Disk | ~20 GB free |
| Keys | ElevenLabs API key (paid plan), OpenAI API key |

**On a MacBook (M1/M2/M3/M4)?** Read [MacBook (Apple Silicon)](#macbook-apple-silicon) first.

### Windows users (WSL2)
1. In PowerShell as admin: `wsl --install -d Ubuntu-24.04`, then reboot.
2. Install the normal **Windows** NVIDIA driver. Do not install a Linux GPU driver inside WSL.
3. Open the Ubuntu terminal and check that `nvidia-smi` prints your GPU.
4. Put this folder inside the Linux filesystem (e.g. `~/voiceforge`), not under `/mnt/c` — it is much faster.
5. The dashboard at `http://127.0.0.1:8765` opens fine from your Windows browser.

## MacBook (Apple Silicon)

Stages 1–7 (writing sentences, ElevenLabs audio, checks and dataset) only call web APIs, so they run well on an M1. Training is the part that needs raw GPU power. Piper's trainer is built for NVIDIA CUDA; Apple's GPU backend (`mps`) is not officially supported by it. Even where it runs, it is many times slower than an NVIDIA card. So there are two routes.

### Route A (recommended): Mac for the data, a rented GPU for training

The pipeline handles the switch for you. When it reaches training and finds no NVIDIA GPU, it:
1. finishes the dataset,
2. packs code + project state into `output/handoff/voiceforge-<voice>.tar.gz`,
3. pauses with a message telling you where the file is.

On the GPU machine you unpack it and press **Resume**. It continues exactly at training, then runs testing and the improvement rounds there.

**1. On the Mac: set up and run**

```bash
cd ~/voiceforge
chmod +x setup_mac.sh start.sh
./setup_mac.sh          # Homebrew, Python 3.11, dashboard env (needs Xcode tools + Homebrew)
./start.sh              # open http://127.0.0.1:8765, fill Settings, press Start
```
Leave Training device on **auto**. Keep the Mac awake while it generates, e.g. `caffeinate -i ./start.sh` in place of `./start.sh`. It takes an hour or two, mostly waiting on ElevenLabs.

**2. Rent a GPU**

Use any provider that gives you an Ubuntu machine with an NVIDIA card and SSH access, such as RunPod, Vast.ai or Lambda. Pick:
- an RTX 3090 / 4090 / A5000 or similar with 24 GB,
- an Ubuntu 22.04 or PyTorch template,
- 50 GB of disk,
- your Mac's SSH public key (`cat ~/.ssh/id_ed25519.pub`, or create one with `ssh-keygen`).

You pay by the hour. A default run (about half a day of training) usually costs a few dollars to low tens, depending on the card.

**3. Copy the project up and open a tunnel to the dashboard**

```bash
# on the Mac (use the host/port your provider shows; -P / -p for a custom SSH port)
scp output/handoff/voiceforge-*.tar.gz root@GPU_HOST:~/
ssh -L 8765:127.0.0.1:8765 root@GPU_HOST
```

**4. On the GPU machine (inside that SSH session)**

```bash
tar xzf voiceforge-*.tar.gz && cd voiceforge
chmod +x setup.sh start.sh && ./setup.sh     # ~10–15 min; ends with "CUDA available: True"
tmux new -s vf                               # keeps running if SSH drops (reattach: tmux attach -t vf)
./start.sh
```
Now open **http://127.0.0.1:8765 on your Mac**; the tunnel forwards it. Press **Resume**. You can close the laptop. To check in later, reconnect with the same `ssh -L …` command and reload the page.

**5. Bring the voice home, then stop the machine**

```bash
# on the Mac
mkdir -p output/final
scp 'root@GPU_HOST:~/voiceforge/output/final/*' output/final/
```
Then **stop or delete the rented machine** in the provider's console. Billing continues until you do.

The handoff archive contains `work/config.json` with your API keys, because testing and improvement rounds still call ElevenLabs and OpenAI. Only upload it to a machine you control, and delete the machine afterwards.

### Route B (experimental): train on the M1 itself

```bash
./setup_mac.sh --with-trainer
```
Then in Settings → Training set **Training device = mps** and **Batch size = 4–8**. Before committing to a full run, test it: set target minutes to 10 and first-round epochs to 20, and check the dashboard's time-per-epoch. Several things can go wrong on Apple GPUs:
- operations that fall back to the CPU (enabled automatically with `PYTORCH_ENABLE_MPS_FALLBACK=1`),
- memory pressure on 8/16 GB machines,
- trainer builds that don't support macOS at all.

Treat it as a way to see the whole loop working, not as the way to train the final voice. A full 1,000-epoch run on an M1 would take days. If setup or training fails on the Mac, switch Training device back to **auto** and use Route A.

### Using the finished voice on your Mac

Piper itself runs fast on Apple Silicon; only training is slow.
```bash
python3 -m venv ~/piper-run && ~/piper-run/bin/pip install piper-tts
~/piper-run/bin/python -m piper -m output/final/en_US-myvoice-medium.onnx -f hello.wav -- "Hello from my own voice."
afplay hello.wav
```

## Setup on Linux / WSL2 (once)

```bash
cd voiceforge
chmod +x setup.sh start.sh
./setup.sh
```

`setup.sh` (Linux) installs and prepares:

| What | Where | Why |
|---|---|---|
| apt: `python3-venv python3-dev build-essential cmake ninja-build git wget espeak-ng` | system | build tools + phonemizer |
| Python env with `fastapi uvicorn requests numpy regex` | `.venv/` | dashboard and pipeline |
| **github.com/OHF-voice/piper1-gpl** (the maintained Piper; `rhasspy/piper` is archived) | `third_party/piper1-gpl/` | trainer, ONNX export, synthesis |
| Its own env with `pip install -e '.[train]'` (PyTorch, Lightning…) + `build_monotonic_align.sh` | `third_party/piper1-gpl/.venv/` | training |
| Lessac **medium** checkpoint from `huggingface.co/datasets/rhasspy/piper-checkpoints` | `checkpoints/base-medium.ckpt` | fine-tuning starting point |

At the end it prints `CUDA available: True` and your GPU name. If it says `False`, see Troubleshooting before you start.

## Run

```bash
./start.sh
```
Open **http://127.0.0.1:8765** → Settings:

1. Paste both API keys → **Save settings**.
2. Press **Load** next to Voice, pick your ElevenLabs voice → **Save settings**.
3. Optionally describe what the voice will read ("smart-home assistant, Indian names and places"), set the language and the target minutes.
4. Press **Start**. That's it.

To keep it running after closing the terminal: `nohup ./start.sh > server.log 2>&1 &`, or run it inside `tmux`.

## What you'll see
- **Stage track**: the ten stages and where it is.
- **Progress bar** with time remaining for the current stage (sentences, clips, epochs).
- **Metrics**: approved audio vs target, approved/rejected clips, ElevenLabs credits used/left, OpenAI tokens, round.
- **Training**: epoch, loss values, progress.
- **Voice quality by round**: word error of your Piper voice on held-out sentences vs the ElevenLabs original, the words it gets wrong, and ElevenLabs-vs-Piper audio players for every test sentence.
- **Clips**: listen to any clip; optionally override approve/reject.
- **Activity log** (also saved to `work/pipeline.log`).

## More expression and emotion

Piper voices have no emotion switch; they learn intonation from the training clips. The dataset is recorded with
steady ElevenLabs settings on purpose, so the voice comes out calm. Two ways to liven it up:

1. **No retraining, instant:** in the dashboard's *Try your voice* box, raise *Expressiveness* (noise, e.g. 0.8–0.9)
   and *Rhythm variation* (noise w, e.g. 1.0–1.2), listen, then *Save as the voice's defaults*. That writes them into
   the `.onnx.json`, so every Piper app uses them. Punctuation matters too: `!`, `?`, commas and `...` all change delivery.
2. **Expressive fine-tune:** *Make it more expressive* writes emotional sentences with OpenAI, records only those
   with lower ElevenLabs stability, and fine-tunes the current voice on old + new clips (one improvement round). Nothing
   already recorded is regenerated. 20–30 minutes of expressive audio is a good amount.

## Output
```
output/final/<voice>.onnx + .onnx.json   ← your finished voice
output/latest/                           ← newest round
output/round_N/                          ← every round kept
work/dataset/                            ← wav/ + metadata_piper1.csv + metadata.csv
```
Use it:
```bash
third_party/piper1-gpl/.venv/bin/python -m piper -m output/final/en_US-myvoice-medium.onnx -f hello.wav -- "Hello from my own voice."
```
Or `pip install piper-tts` anywhere and point it at the two files.

## Costs and time (defaults: 90 minutes target)
- **ElevenLabs:** about 90–110k credits with Multilingual v2. This covers ~76k characters of script, retries, and test references. Flash/Turbo models cost about half. Set **Credit cap** in Settings to hard-limit spending. The run pauses before it would exceed your remaining credits.
- **Transcription checks:** about 2 hours of audio with Scribe v2 or OpenAI transcription, which is cheap.
- **OpenAI sentence writing:** a few hundred thousand tokens. That's cents to a few dollars depending on model.
- **Training:** 1,000 epochs on ~90 minutes of audio is roughly half a day on an RTX 3090/4090-class GPU, and longer on 8–12 GB cards. Each improvement round adds 300 epochs.

## Tuning (Settings)
- **Target approved audio.** 60–120 min is best. For a quick first test, try 20 min with 300 epochs.
- **Max word error per clip** (0.15) and **takes per sentence** (3) control how strict the dataset is.
- **Maximum rounds / stop below word error** control the improvement loop.
- **Restart from this stage** (Settings, bottom) re-runs from any point, e.g. after changing thresholds.
- Other languages: set Language code (`hi`, `mr`…), "Write sentences in" (e.g. `Hindi in Devanagari script`), and eSpeak voice (`hi`, `mr`). Fine-tuning from the English checkpoint is normal.

## Disk space

Each training checkpoint is roughly 800 MB. Voice Forge keeps only the checkpoint it resumes from plus the new run's
best and `last.ckpt`, and deletes old run folders, stale caches and the handoff archive (once this machine can
train) every time training starts. Plan for about 5 GB free during training. To see what uses space:
`du -sh work/* output/* checkpoints third_party 2>/dev/null | sort -h`

## Troubleshooting
- **`CUDA available: False`.** Check `nvidia-smi` first. If it works but PyTorch can't see the GPU, reinstall PyTorch for your CUDA version inside the trainer env:
  `third_party/piper1-gpl/.venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`
- **Out of GPU memory.** Set Batch size to 8 or 12 in Settings, then Resume.
- **`Weights only load failed` or `does not accept option model.sample_bytes`.** These come from *resuming* an old-format checkpoint with new PyTorch. Voice Forge now *warm-starts* from the base checkpoint instead (`--model.warmstart_ckpt`), which avoids both. If you still see them, make sure `pipeline.py` is from the latest zip.
- **`No space left on device`.** Free a few GB (see Disk space), then press Resume. An unreadable, half-written checkpoint is detected and skipped automatically.
- **429 / rate limit.** Lower Parallel ElevenLabs requests to your plan's concurrency.
- **OpenAI model not found.** Put any chat model your key can use in Settings, e.g. `gpt-5-mini`.
- **Mac: `brew: command not found`.** Install Homebrew from brew.sh, then open a new terminal.
- **Mac: `xcode-select` prompt.** Let the Command Line Tools install finish, then rerun `./setup_mac.sh`.
- **Mac: "packed the project" message.** This is expected on route A; follow the MacBook steps above.
- **Cloud GPU: `sudo: command not found`.** `setup.sh` detects root and skips sudo automatically.
- **Start over completely.** Stop the server, delete `work/` and `output/`.

## Files
```
server.py        dashboard + API (127.0.0.1 only)
pipeline.py      all stages, resumable state in work/project.db
textproc.py      cleaning, number spelling, balanced selection, word error
audioqc.py       silence/clipping/pause/rate checks, trimming, loudness
apis.py          ElevenLabs + OpenAI clients with retries
piper_synth.py   runs inside the trainer env to render test sentences
train_launcher.py  starts Piper training (drops the val_mos checkpoint that breaks when MOS is off)
export_launcher.py exports ONNX with the classic exporter Piper was built for
setup.sh         Linux / WSL2 / cloud GPU setup
setup_mac.sh     macOS setup (add --with-trainer for experimental mps training)
dashboard.html   the monitoring page
```

Check that your ElevenLabs plan and terms allow using generated audio to train another model, and that you have rights to the voice.
