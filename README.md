# KDictate

Local speech dictation for KDE Plasma Wayland via IBus.

KDictate is architecturally distinct from generic `whisper-copy-paste`-style tools: it
solves text placement correctly on **KDE Plasma Wayland** by going through IBus and
the KWin input-method protocol, rather than relying on synthetic-keystroke injectors
or clipboard hacks. The `K`-prefix advertises the target environment.

Two desktop components talk over session D-Bus:

1. **Core dictation daemon** (`kdictate-daemon` / `python -m kdictate.core`): persistent mic
   capture/transcribe worker that publishes transcript/state events on the
   session D-Bus. GPU transcription runs in its resident native model worker.
2. **IBus frontend** (`ibus-engine-kdictate` / `python -m kdictate.ibus_engine`): the only component
   allowed to place text into applications; it consumes daemon transcript
   events and maps them to IBus preedit and commit.

Choose Whisper large-v3-turbo on CPU or GPU, or Parakeet TDT 0.6B v3 on GPU.
The 1.0 release candidate keeps GPU models resident for faster responses,
using the pinned whisper.cpp 1.9.5 stack with Vulkan (Q8_0).
The installer detects GPU availability and prompts for one model; Whisper is
the default. See [GPU options and benchmarks](docs/gpu-mode.md).

> KWin Wayland + IBus is the supported configuration. On GNOME or other Wayland
> compositors the daemon + IBus engine path *may* work but is untested.

## Quick start

```bash
cd kdictate
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# optional, for local test/dev tooling such as pytest
pip install -r requirements-dev.txt
```

### System dictation daemon

On Arch/Manjaro, `install.py` handles everything — dependencies, venv, model
selected-model download from Hugging Face, systemd/D-Bus/IBus
registration, and KDE integration:

```bash
python3 install.py
```

It sets up the Python environment, downloads the model, registers the
`io.github.pizzimenti.KDictate.service` systemd user unit, installs the D-Bus
activation file, places the IBus component metadata under the current user's data
directory, writes `~/.config/environment.d/60-kdictate-ibus.conf` so IBus can
scan that per-user component directory, keeps `XMODIFIERS=@im=ibus` for
XWayland/X11 compatibility, installs
`~/.config/plasma-workspace/env/kdictate-plasma-wayland.sh` to unset
`GTK_IM_MODULE` and `QT_IM_MODULE` during Plasma Wayland startup, installs the
project into `~/.local/share/kdictate/.venv` as an editable package so the
IBus component can execute `~/.local/share/kdictate/.venv/bin/ibus-engine-kdictate`,
installs a hidden KDE launcher that calls `gdbus` directly to toggle dictation, configures
Plasma's KWin Wayland input method in `~/.config/kwinrc` to use the installed
`IBus Wayland` desktop file, refreshes the IBus cache, and restarts
`ibus-daemon` for the current session.

To start the daemon manually instead:

```bash
source .venv/bin/activate
python -m kdictate.core
```

The daemon publishes the reverse-DNS session bus name
`io.github.pizzimenti.KDictate1`, and the IBus frontend is expected to be
selected by that engine name once the frontend package is installed.

The core service is user-level and idempotent:

```bash
systemctl --user enable --now io.github.pizzimenti.KDictate.service
```

If you want to confirm the D-Bus API manually, use the terminal control path
once the daemon-side service is available:

```bash
python -m kdictate.cli start
# speak
python -m kdictate.cli stop
```

`stop` waits for transcription to finish and prints the latest transcript from
the daemon runtime files.

### IBus selection flow

After the IBus frontend is installed, enable it the same way you would any other IBus engine:

1. Open your IBus configuration tool.
2. Add `KDictate` or the reverse-DNS engine name `io.github.pizzimenti.KDictate1`.
3. Select it in the input method switcher when you want dictation text to flow
   into the focused application.

The daemon never inserts text directly. Partial transcript should appear as
preedit and final transcript should be committed by the IBus frontend only.
If the engine still does not appear immediately after install, or if text
fields do not accept dictation commits, sign out and back in once so the
desktop session reloads the updated input-method environment and KWin picks up
the configured IBus Wayland input method. On Plasma Wayland, `GTK_IM_MODULE`
and `QT_IM_MODULE` should remain unset in the desktop session; the
compositor-backed `IBus Wayland` path handles native Wayland clients.

On KDE Plasma, `io.github.pizzimenti.KDictateToggle.desktop` can be bound as a
global shortcut. Its `Exec=` line calls `gdbus` directly against the session
D-Bus, bypassing Python startup for the toggle hot path — this makes
`Ctrl+Space` noticeably snappier than a Python-wrapped toggle.

## Architecture

- `kdictate/core/daemon.py` (`python -m kdictate.core`): long-lived daemon that
  keeps the Whisper model warm, owns microphone capture, VAD segmentation,
  transcription, and publishes state/transcript events over session D-Bus.
- `kdictate/ibus_engine/` (`python -m kdictate.ibus_engine`): IBus engine
  process that subscribes to daemon D-Bus signals and maps them to IBus
  preedit/commit. This is the **only** path allowed to place text into
  applications.
- `kdictate/cli/dictatectl.py` (`python -m kdictate.cli`): stdlib D-Bus control
  plane for `start`, `stop`, `toggle`, `status`, and `last-text`. Use this from
  a terminal for scripting. For the `Ctrl+Space` hot path, the installed
  `.desktop` file calls `gdbus` directly instead.
- `kdictate/`: package providing constants, exceptions, logging, D-Bus API
  definition, and the service/IBus/CLI subpackages.

### Runtime files

The daemon and helpers coordinate through two files under `XDG_RUNTIME_DIR`:

- `kdictate-<uid>.state`: current daemon state (`idle`, `starting`, `recording`, `transcribing`, or `error`)
- `kdictate-<uid>.last.txt`: latest completed transcript

The daemon owns writes to those files. The CLI reads them so control/status
behavior stays consistent across shells and user services.

## GPU mode (optional)

KDictate supports Whisper and Parakeet through Vulkan. GPU models stay
loaded between utterances to reduce response time. No CUDA or ROCm stack
is required; a working Vulkan driver is needed.

### Setup

GPU support ships **with the package**: the kdictate package vendors a
pinned, Vulkan-enabled `whisper.cpp` (with `ggml` bundled) under
`/usr/lib/kdictate/`.  There is nothing extra to install and no dependency
on the fast-moving `whisper.cpp-vulkan` / `llama.cpp-vulkan` AUR packages —
the stack is version-locked to the kdictate release.  See
`docs/packaging.md` for the rationale.

1. Build and install the package (it carries the GPU stack):

   ```bash
   # Arch / Manjaro
   cd packaging && makepkg -si
   ```

   You also need a working Vulkan driver for your GPU — e.g.
   `vulkan-radeon`, `vulkan-intel`, or `nvidia-utils`.

2. Run the installer — it detects GPU availability and prompts:

   ```bash
   python3 install.py
   ```

   The installer downloads the selected model: Whisper GPU (~874 MB),
   Parakeet GPU (~669 MB), or Whisper CPU (~1.6 GB). It configures the
   service with `--backend gpu`, `--backend parakeet`, or `--backend cpu`.

> **Running from a source checkout?** The package isn't installed, so build
> the same pinned whisper locally and point kdictate at it:
>
> ```bash
> ./packaging/build-whisper.sh
> export KDICTATE_STT_WORKER="$PWD/.whisper/build/bin/kdictate-stt-worker"
> ```

See `docs/gpu-mode.md` for architecture details and benchmark results.

## Tuning

- `--backend cpu|gpu|parakeet`: transcription backend, fixed at install time (default: cpu; no runtime fallback).
- `--cpu-threads N`: override thread count. Dictation-oriented defaults now use physical cores / short-form-friendly thread counts.
- `--compute-type int8|float16|float32`: precision/runtime tradeoff.
- `--language`: defaults to `en`.
- `--beam-size`: daemon and live CLI default to 1.
- `--state-file`: daemon runtime state file path (default: `$XDG_RUNTIME_DIR/kdictate-<uid>.state`).
- `--last-text-file`: latest transcript cache path (default: `$XDG_RUNTIME_DIR/kdictate-<uid>.last.txt`).
- `--vad-filter/--no-vad-filter`: daemon defaults to `vad_filter=True` to catch silence before the encoder.
- `--condition-on-previous-text/--no-condition-on-previous-text`: daemon defaults to `False` to reduce cascading hallucinations.
- `--no-speech-threshold`: Whisper-side non-speech rejection. The daemon defaults to `0.6`.
- `--energy-threshold`, `--start-speech-ms`, `--silence-ms`, `--max-utterance-s`: live CLI utterance-boundary controls.
- `--task transcribe|translate`: keep original language vs force English output (CLI mode only).
- `--decode-workers`, `--diag`, `--diag-interval-s`: parallelism and diagnostics (CLI mode only).
- Runtime control/VAD polling now uses 150ms wait intervals to reduce idle wakeups without materially affecting dictation latency.

## Files

- `install.py`: install dependencies, register the user service, and install the D-Bus and IBus metadata (Arch/Manjaro).
- `pyproject.toml`: package metadata and console-script entry points.
- `requirements-dev.txt`: local development and test-only dependencies.
- `kdictate/`: core package — D-Bus contract, daemon logic, IBus frontend, CLI, runtime utilities, audio helpers (`kdictate.audio_common`), and CPU thread / compute-type selection (`kdictate.runtime_profile`).
- `packaging/kdictate-systemd.service`: systemd user unit for the core daemon (`ExecStart=... kdictate-daemon`).
- `packaging/io.github.pizzimenti.KDictate.service`: D-Bus session activation file (delegates to the systemd unit via `SystemdService=`).
- `packaging/io.github.pizzimenti.KDictate.xml`: D-Bus introspection XML published on the session bus.
- `packaging/io.github.pizzimenti.KDictate.component.xml`: IBus component metadata for the engine frontend.
- `packaging/io.github.pizzimenti.KDictateToggle.desktop`: hidden KDE application entry that binds `Ctrl+Space` to a direct `gdbus call` against the session bus.
- `packaging/60-kdictate-ibus.conf`: `environment.d` snippet that adds the per-user IBus component directory to `IBUS_COMPONENT_PATH` and sets `XMODIFIERS=@im=ibus`.
- `packaging/kdictate-plasma-wayland.sh`: Plasma session env script that unsets `GTK_IM_MODULE` and `QT_IM_MODULE` to let the compositor-backed IBus Wayland path handle native clients.
- `check_ibus_only.py`: regression check for forbidden injector and clipboard backends.

## Notes

- The standalone daemon defaults to CPU; the installer defaults to Whisper GPU when available.
- The selected model is downloaded to `~/.local/share/kdictate/`. Existing GPU models are reused after SHA-256 verification.
