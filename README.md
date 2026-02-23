# whisper-cli (CPU-First, Laptop-Optimized)

This project runs **high-accuracy Whisper transcription** from your **default system microphone** and prints text directly to the terminal in near real time.

No audio files are read for transcription input and no transcript files are written by default.

It uses:
- `distil-whisper/distil-large-v3` as the default source model
- CTranslate2 conversion + int8 quantization for faster local inference
- `faster-whisper` for efficient transcription runtime

## Why this setup

On this machine type (no NVIDIA CUDA GPU detected), the best practical local path is:
1. Convert the model to CTranslate2 format.
2. Run inference on CPU with `int8`.
3. Default to a larger distilled model for better accuracy.
4. Keep runtime CPU-only for predictable local behavior.

## Quick start

```bash
cd whisper-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` is configured to use a CPU-only PyTorch wheel (no CUDA runtime packages).

Convert the model once:

```bash
python prepare_model.py
```

If `torch` is unavailable for your Python version, create a Python 3.12 venv for conversion.

Run live mic transcription:

```bash
python mic_realtime.py
```

Press `Enter` to stop.

`mic_realtime.py` currently uses a simple baseline speech gate and stock-like decode settings for testing.

## Files

- `prepare_model.py`: downloads/converts the default model to local CTranslate2 format.
- `mic_realtime.py`: captures default microphone input and prints live text to terminal.
- `transcribe.py`: transcribes a single file with laptop-tuned defaults.
- `benchmark.py`: measures latency and real-time factor across multiple runs.
- `runtime_profile.py`: hardware-aware runtime defaults.

## Tuning knobs

- `--cpu-threads N`: override thread count.
- `--compute-type int8|float16|float32`: precision/runtime tradeoff.
- `--task transcribe|translate`: keep original language vs force English translation.
- `--language`: defaults to `en` for no-flag runs.
- `--energy-threshold`: raise this if room noise is triggering false speech.
- `--silence-ms`: lower for faster phrase commits, higher for longer phrases.
- `--max-utterance-s`: cap utterance length before forced decode.
- `--beam-size`: defaults to 5 (stock), lower is faster.
- `--decode-workers`: parallel decode workers (higher can increase CPU utilization).
- `--diag` / `--diag-interval-s`: periodic runtime diagnostics to stderr for bottleneck analysis.

## Notes

- First conversion/download can take time and several GB of storage.
- This project intentionally runs CPU-only.
- Live mode does not create transcript files unless you explicitly redirect terminal output.
