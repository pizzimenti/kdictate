# KDictate GPU Mode

Optional Vulkan-accelerated transcription via whisper.cpp.  Preserves
the zero-dependency CPU default while cutting decode latency roughly
in half on systems with a Vulkan-capable GPU.

## How it works

The daemon supports two transcription backends behind a common
`TranscriptionBackend` protocol:

- **CPU** (default): faster-whisper / CTranslate2, int8 quantization.
  No extra dependencies.  ~5 s decode floor on a Ryzen 5 8640HS.
- **GPU** (opt-in): whisper.cpp CLI with Vulkan, Q8_0 quantization,
  beam 3, flash attention.  ~2.5 s decode on the same hardware.

Both use the same large-v3-turbo model weights.  The GPU path uses
Q8_0 rather than FP16 because benchmarking showed it is 15 % faster
with no measurable accuracy loss, even under heavy background noise
(SNR 5 dB).  Beam 3 is free on the GPU (the encoder dominates) and
preserves capitalization and punctuation that beam 1 sometimes drops.

Backend selection is controlled by `--backend cpu|gpu`:

- `cpu` — use faster-whisper (default, no GPU needed).
- `gpu` — require whisper.cpp + Vulkan; fail if unavailable.

The backend is chosen **once, at install time** — exactly one, never both.
The installer detects GPU availability, prompts, downloads only that
backend's model, and pins the choice (a packaged install via a systemd
drop-in, a source install by baking the flag into its unit). There is no
`auto`/runtime fallback: the other backend's model isn't provisioned, so a
fallback would fail anyway — and a silent 2× slowdown is worse than a clear
failure you can act on.

## Requirements for GPU mode

GPU mode ships vendored with the kdictate package: a pinned,
Vulkan-enabled `whisper-cli` is installed under `/usr/lib/kdictate/bin`
with `ggml` bundled in — there is **no** dependency on `whisper.cpp-vulkan`
or `llama.cpp-vulkan`.  See `docs/packaging.md` for why, and how to bump
the pinned version.

- The vendored `whisper-cli` — shipped by the package, or for a source
  checkout built via `./packaging/build-whisper.sh` and selected with the
  `$KDICTATE_WHISPER_CLI` environment variable
- The GGML Q8_0 model (~874 MB), downloaded automatically by the installer
- A Vulkan-capable GPU with a working driver

The daemon resolves the binary in this order: `$KDICTATE_WHISPER_CLI`, then
the vendored `/usr/lib/kdictate/bin/whisper-cli`, then a `whisper-cli` /
`whisper-cpp` on `PATH` (see `find_whisper_cpp` in `backend.py`).

## Architecture

Nothing outside `backend.py` knows which backend is active.  The VAD
segmenter, D-Bus service, IBus engine, and CLI are unchanged.

```text
daemon.py
  └─ TranscriptionBackend.transcribe(pcm_chunks, audio_seconds) -> str
       ├─ FasterWhisperBackend  (CPU, delegates to transcribe_pcm)
       └─ WhisperCppBackend     (GPU, subprocess to whisper-cli)
```

At startup the daemon probes the GPU backend by feeding 1 s of silence
to whisper.cpp.  If the probe fails (missing binary, missing model,
Vulkan driver error), the daemon falls back to CPU with a log message.
