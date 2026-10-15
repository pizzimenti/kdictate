# KDictate GPU mode (1.0)

Choose Whisper GPU, Whisper CPU, or Parakeet GPU in `python3 install.py`.
Whisper remains the installer default when Vulkan is available. The installer
provisions the selected model and pins the service to that choice. Existing
`--backend gpu` configurations continue to mean Whisper.

| Choice | CLI flag | Model | Download |
| --- | --- | --- | --- |
| Whisper GPU | `--backend gpu` | Whisper large-v3-turbo Q8_0 | 874 MB |
| Parakeet GPU | `--backend parakeet` | Parakeet TDT 0.6B v3 Q8_0 | 669 MB |
| Whisper CPU | `--backend cpu` | Whisper large-v3-turbo CTranslate2 int8 | about 1.6 GB |

Parakeet v3 recognizes supported languages automatically; Whisper accepts
`--language`. Whisper's beam and no-speech settings do not tune Parakeet.
Parakeet is a separate speech model, so names, punctuation and
recognition mistakes can differ. Compare it with recordings from your own
microphone before choosing it for daily dictation. These GPU choices require a
working Vulkan driver. A failure is reported for the selected engine.

## Resident models

The daemon starts `kdictate-stt-worker`, loads the selected model, and warms its
decode path with one second of silence before accepting requests. The worker
keeps model weights and compute buffers allocated between recordings. Stopping
recording releases the microphone while the model remains ready. Stopping the
daemon releases the worker and its memory.

The private worker accepts length-prefixed 16 kHz mono PCM through stdin and
returns length-prefixed UTF-8 through stdout. It opens no network port. Each
utterance resets transcript context. Timeouts, malformed replies and worker
crashes fail the current transcription and reap the worker. A later request can
restart the same engine; failed audio is never replayed automatically.

The CPU backend already keeps its faster-whisper model in the daemon. GPU
residency brings the same lifecycle to Whisper and Parakeet, at the cost of
retaining their memory while dictation is idle.

## Native stack and source installs

The package vendors whisper.cpp 1.9.5, its Parakeet implementation, bundled
GGML, and Vulkan/CPU modules under `/usr/lib/kdictate`. It has no dependency
on a separately updated llama.cpp or whisper.cpp package. Models are downloaded
separately from pinned Hugging Face revisions and verified with SHA-256.
See [packaging](packaging.md) for build details.

For a source checkout:

```bash
./packaging/build-whisper.sh
export KDICTATE_STT_WORKER="$PWD/.whisper/build/bin/kdictate-stt-worker"
python3 install.py
```

The source installer also discovers that local build automatically and copies
its worker and shared libraries into `~/.local/share/kdictate/native/bin`.
Runtime resolution is the explicit `KDICTATE_STT_WORKER` override, packaged
worker, per-user worker, then `PATH`.

## Reproducing the comparison

The [2026-10-15 report](../benchmarks/results/2026-10-15/report.html) includes
all transcripts, timings, startup and memory measurements, idle tests and raw
JSON. On this Radeon 760M host, resident Whisper decoded at a 1.75s median
and resident Parakeet at 0.17s, compared with 2.20s for the existing Whisper
CLI. WER was 2.59%, 2.96% and 2.69%, respectively. The engine update alone
showed a small, inconsistent timing difference between passes. Both resident
adapters matched their corresponding CLI transcripts exactly on all 62 clips.
Retained GPU allocations after a long clip were about 1,184 MiB for Whisper
and 790 MiB for Parakeet; process RSS alone omits most of that shared memory.

Large data and build artifacts belong in the disk-backed
`~/.cache/kdictate-benchmarks` directory. The benchmark uses human LibriSpeech
test-clean and test-other audio with published transcripts. The preparation
script selects 30 uniformly spaced rows from each pinned split, verifies the
source archives, and writes a manifest containing per-clip audio hashes. Two
digital silence clips are scored separately for unwanted text.
Audio paths are relative to the generated manifest, so the audio directory can
be moved between machines. The checked-in manifest records the corpus; generate
the WAV files with the preparation script and pass its generated manifest to
the runner. Historical results retain their original manifest hash alongside
the normalized hash after path-only portability edits.

```bash
.venv/bin/pip install -c constraints.txt -r requirements-benchmarks.txt
.venv/bin/python benchmarks/prepare_audio.py
.venv/bin/python benchmarks/run.py \
  --manifest "$HOME/.cache/kdictate-benchmarks/audio/manifest.json" \
  --binary .whisper/build/bin/kdictate-stt-worker \
  --model "$HOME/.local/share/kdictate/ggml-large-v3-turbo-q8_0.bin" \
  --mode resident --engine whisper --name 'Whisper resident' \
  --output "$HOME/.cache/kdictate-benchmarks/results/whisper-resident.json"
```

Use `--mode cli` with the appropriate CLI for a fresh model process per clip.
Parakeet uses `--engine parakeet` with its own model path. Run configurations
serially without concurrent builds. Resident tests measure load and warmup
separately from utterance latency; CLI timing includes process/model startup.
The operating system's file cache is warm, so these are process-start times,
not cold-boot disk-loading measurements. Memory figures are Linux process RSS;
GPU/shared memory accounting can differ.

WER is total word insertions, deletions and substitutions divided by total
reference words. Case and punctuation are ignored; number formatting and
contractions remain literal. Decode failures count as empty hypotheses and are
also reported explicitly. Read audiobooks do not represent every accent,
background noise, short command, technical term or microphone used for dictation.
The benchmark excludes capture, silence detection and desktop text insertion.

`benchmarks/compare.py` runs all five configurations serially and generates
the HTML/CSV report. Supply `--manifest`, `--old-cli`, `--new-bin-dir`,
`--whisper-model`, `--parakeet-model` and `--output-dir`. Preserve an old CLI
and its matching private libraries when comparing across installed releases.
`benchmarks/check_responsiveness.py` accepts the same inputs with `--output`
instead of `--output-dir`; it counterbalances eight CLI pairs, then checks
each resident engine after 0, 10 and 30 seconds idle.

The optional daemon smoke check runs under `dbus-run-session` with a 30-second
external timeout, `--engine whisper|parakeet`, `--worker`, `--model`, and
`--output-dir`. It exercises real model readiness, D-Bus service startup and
worker reaping without opening the microphone or using the desktop session bus.
