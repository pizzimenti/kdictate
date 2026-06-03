# Packaging: the vendored whisper.cpp GPU stack

KDictate's GPU backend shells out to a `whisper-cli` binary. How that
binary gets onto the system is a deliberate, slightly unusual choice, and
this document explains why — so a future maintainer doesn't "simplify" it
back into the problem it solves.

## The problem we're avoiding

The obvious way to get GPU transcription on Arch is `yay -S
whisper.cpp-vulkan`. That package builds with `-DWHISPER_USE_SYSTEM_GGML=ON`,
so it depends on `llama.cpp-vulkan` to provide the shared `ggml` library.
Two facts make that a poor foundation to ship on:

1. **llama.cpp has no semantic versioning.** Upstream tags a new `b####`
   build essentially every day (often several times a day) — `b####` is a
   monotonic build counter, not a `major.minor.patch` with compatibility
   guarantees. There is no stable/LTS branch; `master` *is* the release.
   So `llama.cpp-vulkan` churns daily, and every bump is a full local
   recompile of an LLM inference engine we never invoke.

2. **`ggml`'s ABI is not stable between tags.** `ggml` is developed
   in-lockstep inside the llama.cpp tree, and symbols come and go freely
   between `b####` tags. If the installed `ggml` and the whisper build
   drift apart, whisper fails to compile or link — e.g. whisper 1.8.6's
   `talk-llama` example calling `ggml_backend_meta_device`, a symbol absent
   from an older pinned `ggml`. That exact skew is what motivated this
   change.

For software we *ship*, neither "recompile daily" nor "depends on a
fast-moving library with an unstable ABI" is acceptable. A user-side pin
(`IgnorePkg`) is not a fix either — it doesn't travel with the software
and no end user should have to know about it.

## The decision

**Vendor the GPU stack into the kdictate package, version-locked.** The
package builds its own `whisper.cpp` with `ggml` bundled
(`-DWHISPER_USE_SYSTEM_GGML=OFF`) into the private prefix
`/usr/lib/kdictate/`, with **no** dependency on `whisper.cpp-vulkan` or
`llama.cpp-vulkan`. The whisper version is pinned by a single variable,
`_whisper_ver` in `packaging/PKGBUILD`, and only moves when we cut a
kdictate release.

The result: no daily churn, no `ggml`-ABI skew (one package owns both
sides), and the transcription stack is reproducibly tied to each kdictate
release.

### Build flags and why each one

From `packaging/PKGBUILD`:

| Flag | Why |
| :--- | :--- |
| `WHISPER_USE_SYSTEM_GGML=OFF` | Compile whisper's own pinned `ggml`; removes the `llama.cpp-vulkan` dependency and the ABI skew entirely |
| `GGML_VULKAN=ON` | The GPU decode path |
| `GGML_NATIVE=OFF` | Do **not** bake the build host's `-march=native`; an AVX-512 binary built on Zen 4 would `SIGILL` on Zen 2/3 |
| `GGML_CPU_ALL_VARIANTS=ON` + `GGML_BACKEND_DL=ON` | Compile baseline/AVX2/AVX-512 CPU codepaths and dispatch at runtime — one package, compiled once, runs portably *and* fast across Ryzen Zen 2 → Zen 5 |
| `WHISPER_SDL2=OFF` | We feed WAV over stdin, so SDL2 is unused. It also gates the `talk-llama` example — the thing whose `ggml`-ABI mismatch caused the original break |
| `WHISPER_FFMPEG=OFF`, `WHISPER_BUILD_SERVER=OFF`, `WHISPER_BUILD_TESTS=OFF` | Unused; trims the build |
| RPATH `$ORIGIN/../lib` | So `whisper-cli` in `…/bin` finds its bundled `libggml-*.so` (incl. the CPU variants and the Vulkan backend) in `…/lib` |

Because `CPU_ALL_VARIANTS` makes the variants explicit, the build host's
own CPU is irrelevant — you can build the package on any machine and it
still runs everywhere.

## Bumping the pinned whisper version

1. Edit `_whisper_ver` in `packaging/PKGBUILD` (and the matching constant
   in `packaging/build-whisper.sh`).
2. Run `updpkgsums` to refresh `sha256sums`.
3. Build (`makepkg`), verify the daemon's GPU probe passes, then bump
   `pkgver` and cut the kdictate release.

This is the whole point: **we** curate the whisper cadence once, for every
user, as part of a release — instead of every machine tracking upstream's
daily firehose.

## Binary resolution (runtime)

`backend.find_whisper_cpp()` resolves the binary most- to least-specific:

1. `$KDICTATE_WHISPER_CLI` — explicit override (dev / custom builds)
2. the vendored `/usr/lib/kdictate/bin/whisper-cli` (packaged installs)
3. a `whisper-cli` / `whisper-cpp` on `PATH` (source/dev fallback)

## Development from a source checkout

A `git clone` has no installed package, so build the same pinned whisper
locally with the dev helper and point kdictate at it:

```bash
./packaging/build-whisper.sh
export KDICTATE_WHISPER_CLI="$PWD/.whisper/build/bin/whisper-cli"
```

The dev build uses `GGML_NATIVE=ON` and a single static binary (it only
runs on your machine, so tuning is free and simplest). The shipped package
uses the portable multi-variant build instead; transcription output is
identical — only CPU dispatch differs.

## Alternatives considered

| Option | Daily rebuilds | Pulls `llama.cpp-vulkan` | ABI skew | One artifact, Zen 2→5 | Whisper bumps only w/ kdictate | Custom tooling |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `yay -S whisper.cpp-vulkan` (status quo) | yes | yes | yes | n/a | no | none, but churny |
| Monthly local-repo pin of `llama.cpp-vulkan` | monthly | yes | **high** (half-pin caused the break) | no | no | heavy/fragile |
| `IgnorePkg` personal pin | no | yes (held) | possible | no | no | user-side, not shippable |
| Source build per machine, `NATIVE=ON` | release-only | no | no | **no** (tied to build CPU) | yes | a PKGBUILD |
| **Vendored multi-variant (this)** | **no** | **no** | **no** | **yes** | **yes** | one static PKGBUILD |
| Flatpak | no | no | no | all distros | yes | manifest + IBus re-architecture |

The vendored multi-variant build is the only option that is simultaneously
churn-free, `llama`-free, skew-proof, fleet-portable from a single compile,
and version-locked to kdictate — without Flatpak's IBus-engine
re-architecture (kdictate registers a host IBus engine, which a sandbox
can't) or the per-machine compiles a `NATIVE=ON` source build forces.

## Caveats

- **We own the whisper update cadence**, including security/bugfixes: a
  whisper fix reaches users only when we re-pin and re-release. That is the
  correct trade for shipped software — the publisher curates so users don't.
- **`GGML_NATIVE=OFF`** gives up per-CPU autovectorization, but
  `CPU_ALL_VARIANTS` recovers most of it via runtime dispatch, and the heavy
  compute runs on the GPU anyway — so the CPU-path delta is small.
- **Stale-Arch edge case:** the binary dynamically links glibc/libstdc++/
  libvulkan. On rolling Arch this is a non-issue; a very out-of-date install
  could in theory hit a glibc-too-old error, and the GPU probe falls back to
  CPU if so.
- The **model weights (~874 MB)** are not vendored — they remain downloaded
  on first run (see follow-ups).

## Python runtime (also vendored)

kdictate's source install (`install.py`) builds a per-user venv at
`~/.local/share/kdictate/.venv`; it never used system Python packages. The
package mirrors that isolation rather than declaring Arch `depends`, because
several deps (`faster-whisper`, `ctranslate2`, `sounddevice`) are AUR-only —
declaring them would force an AUR helper at build time and pin to whatever
the AUR ships instead of `requirements.txt`.

Instead, exactly like the **voiceagent** package:

- `build()` runs `pip install --target …/vendor -r packaging/vendor-requirements.txt --only-binary=:all:`, vendoring the deps as wheels into `/usr/lib/kdictate/vendor`.
- The wheel's console scripts are moved to `/usr/lib/kdictate/libexec/`, and `/usr/bin/{kdictate-daemon,kdictatectl,ibus-engine-kdictate}` become symlinks to `packaging/kdictate-launcher.sh`, which prepends the vendor dir (and system site-packages) to `PYTHONPATH` before exec'ing the real entry point.
- **`PyGObject` is the one exception** — it builds C bindings against the system `gobject-introspection` and isn't a clean `--only-binary` target, so it ships as the `python-gobject` system dependency. `python-gobject` lives in the system site-packages, which the launcher also puts on `PYTHONPATH`.

`packaging/vendor-requirements.txt` mirrors `requirements.txt` minus
PyGObject; keep them in sync when bumping deps.

### Integration templates

The IBus component, systemd unit, and D-Bus service files carry
`@@REPO_DIR@@` / `@@BACKEND_FLAGS@@` / `@@ENGINE_EXEC@@` / `@@HOME@@`
placeholders that `install.py` renders for a per-user venv install. The
package renders them to **system** paths in `package()` (entry points under
`/usr/bin`, component under `/usr/share/ibus/component`, etc.), defaulting
the daemon to `--backend auto`. `@@HOME@@` becomes `${HOME}`, which
`environment.d` expands per-user at session start.

## Packaging follow-ups (not yet done)

The package installs the binary, the Python runtime, and all integration
files — but it is **not yet a complete turn-key install**:

- **Model provisioning.** The ~874 MB GGML/CTranslate2 model is still
  fetched by `install.py` on first run; the package does not ship or fetch
  it. A first-run daemon step (or a `kdictate-model` split package) is
  needed before a package-only install can transcribe.
- **`install.py` vs package coexistence.** `install.py` writes per-user
  files (`~/.local/share/ibus/component`, `~/.config/systemd/user`); the
  package writes system equivalents. Running both double-wires the engine.
  Decide which is canonical and have `install.py` detect a packaged install.
- **No `LICENSE` file** exists in the repo yet, so the `package()` license
  install is skipped — add one (`license=('MIT')` is declared).
- **Service enablement** is left to the user (`systemctl --user enable
  --now io.github.pizzimenti.KDictate.service`), per Arch packaging norms.
