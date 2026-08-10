# Changelog

## 0.15.0

### Fixed

- **The daemon no longer forces the microphone to 91% on every activation.**
  This is the root cause behind the VAD never finding a silence gap. v0.10.0
  pinned capture to 91% because the source had drifted to 40% and speech fell
  below the threshold — which fixed that case and created a worse one: on an
  already-healthy source it drives capture into clipping, and a clipped signal
  has no dynamic range left for an energy VAD to segment. Measured at 91% on
  the author's hardware: `peak_rms` 32768 (int16 maximum, i.e. clipping),
  noise floor ~9000 against a speaking level of ~11000-13000. A 1.3x ratio
  between "room" and "voice" is not separable by *any* threshold, which is why
  1500, 1000, 700 and the adaptive gate all failed in turn.
  `--mic-min-volume-percent` (default 50) is now a floor the daemon rescues
  you from rather than a level it imposes: it only ever raises, only from
  below the floor, and `0` disables the behaviour entirely. **Existing users
  should set their input level in system settings** — the daemon will now
  leave it alone.
- **Short words were silently dropped** when the adaptive gate was enabled.
  Blocks are withheld until the noise floor is measurable, but a recording
  that ended before that point discarded the buffer outright, and with
  `total_blocks` still 0 even the "no speech detected" warning could not fire.
  A push-to-talk word like "yes" fits entirely inside that window. Held blocks
  are now scored against the configured threshold when the recording ends
  early.
- **`install.py` could not detect a source install's version at all.** The
  probe ran `python -c` with the installer's own cwd, so `sys.path[0]` was the
  repo root and it imported the tree being installed *from* rather than the
  installed copy — always reporting this tree's version and making the version
  gate a permanent no-op. Now uses `-P` and a cwd outside the checkout.
- **`install.py --reconfigure`** restores the repair path. Every configuration
  step is idempotent and re-running them is how a clobbered Ctrl+Space
  binding, a missing IBus registration, or a backend switch gets fixed; the
  version gate skipped all of them when versions matched, leaving an install
  broken *at the current version* unfixable short of editing
  `app_metadata.py`. Reconfiguring skips the package rebuild, which has
  nothing to do.
- **A bare `pytest` at the repo root now works.** It recursed into makepkg's
  `packaging/src` staging tree, where setuptools alone ships ~142 test modules
  that cannot be collected there, so the obvious invocation reported "142
  errors during collection" and ran none of this project's tests — on any
  machine where a package had been built, which a packaged install now always
  does. Fixed with `testpaths`/`norecursedirs` rather than per-test skips.

- **Upgrading no longer leaves stale IBus engine processes behind.** An engine
  reads its script once at spawn and keeps executing what it loaded, so
  replacing the package left the running engine on the *previous* version's
  code while IBus spawned a second copy alongside it. Both sit on the session
  bus receiving the daemon's `FinalTranscript` broadcast, but only one can
  hold the focused input context — so a transcript could be delivered to an
  engine unable to commit it, and dictation silently produced no text until
  the next reboot. `refresh_ibus_registry` already killed `ibus-daemon`, but
  with `pkill -x`, an exact *name* match that never reached
  `python …/ibus-engine-kdictate`; killing the daemon orphaned the engines
  rather than reaping them, which is why they survived `ibus restart` too.
  Observed here with an engine that outlived both an upgrade and a restart.
  The kill is scoped to the invoking user and matched on the executed path.

### Changed

- **Removed the adaptive gate's ceiling.** It was `energy_threshold * 8`, and
  `energy_threshold` is a weak-microphone floor that says nothing about the
  signal actually arriving. On hardware with a ~9000 noise floor the ceiling
  sat at 5600 — *below* the noise — so every block scored as voiced and
  utterances ran to `max_utterance_s`, reintroducing the exact failure the
  gate exists to prevent. Unclamped, a loud room can instead push the gate
  above the voice and reject a session: a louder failure, but one the
  end-of-session warning names the knob for, rather than a silent stream of
  Whisper hallucinations over room noise.
- `GateCeilingTest` passed only because it picked an ambient level that
  happened to sit just under the old ceiling. Replaced with `LoudRoomTest`,
  which uses the level that actually demonstrated the failure.

## 0.14.2

### Fixed

- **Dictation produced no text at all.** 0.14.0 added a per-recording counter
  for the session-limit prompt and named it `_session_generation` — a name
  already in use for something else entirely. That counter moves only when
  the daemon rotates its session primitives (wedge recovery, safety-net
  teardown), and `_decode_worker` reads it as "this session is gone, discard
  your transcript". Bumping it on every recording start therefore made every
  normal session look like a wedge recovery: the decode worker exited, the
  transcript was dropped, and every session ended `final transcript emitted
  (0 chars)` / `no speech detected`. The prompt's counter is now
  `_recording_epoch` and the two are fully independent.

### Changed

- **The adaptive noise-floor gate is off by default** (`--noise-floor-margin`
  now defaults to `0`). Measured on real hardware it tracked the speaker
  rather than the room: a push-to-talk session is short and mostly speech, so
  the trailing-window low percentile settled on quiet speech (~9000 RMS
  against an ~11000-13000 speaking level) with no silence to anchor it. The
  gate then either sat above the voice and rejected the session, or — with
  the ceiling anchored to `energy_threshold`, which is a weak-microphone
  floor and unrelated to the observed signal — was clamped below the noise
  and passed every block, restoring the never-ending-utterance bug it was
  added to fix. The mechanism is unchanged and still available; it is simply
  opt-in until it can be tuned against real logs.
- The noise floor is now measured and logged even when the gate is disabled,
  and the `recording ended:` line reports the full `rms=min-max` range and the
  margin in force. `peak_below_gate` collapses to 0 exactly when the gate is
  wrong, which is when the number is needed most, so it is no longer the only
  evidence available.
- The forbidden-injector regression test no longer scans makepkg's
  `packaging/src` and `packaging/pkg` staging trees. They hold extracted
  whisper.cpp sources and pip-vendored wheels — numpy alone contains `wtype`
  in unrelated places — so the guard failed on any machine where a package
  had been built.

## 0.14.1

### Fixed

- **The installer is quiet again.** 0.14.0 added the package rebuild without
  suppressing its output, making it the only one of 18 subprocess calls in
  `install.py` that wasn't quiet — so a rebuild dumped the whole wheel build,
  pip vendoring, and cmake log over the one-screen checklist. Both the build
  and the install now run quiet like every other step, and print the tail of
  the captured output only when they fail, so a real build error is still
  readable without the successful path being a wall of text.
- **No more sudo/pacman prompts in the middle of a step.** Dropped
  `makepkg --syncdeps`, which would start a package transaction inside a step
  whose output was suppressed. Missing build dependencies are now detected up
  front and reported with the exact command to install them, before anything
  is built or changed. `sudo` credentials are likewise requested visibly
  before the install step rather than from inside it.
- Removed the `SetuptoolsDeprecationWarning` at its source: `pyproject.toml`
  declares `license = "MIT"` as an SPDX string rather than a TOML table.

## 0.14.0

### Fixed

- **Dictation typed nothing at all.** The `ibus` CLI locates its bus socket by
  display name (`~/.config/ibus/bus/<machine-id>-unix-wayland-0`), not over
  D-Bus. The daemon is started from `default.target` at login, before the
  session exports `WAYLAND_DISPLAY` into the systemd user environment, so
  every call read a stale socket file from an earlier session and failed with
  `ibus_bus_get_global_engine: assertion 'IBUS_IS_BUS (bus)' failed`. The
  active engine was therefore never switched to KDictate, nothing subscribed
  to `FinalTranscript`, and audio was recorded and transcribed into the void.
  The daemon now recovers the display variables from the systemd user manager
  at call time. This is the failure the v0.11.x "self-heal" was meant to
  catch — its probe was blind for the same reason.
- **The "still recording" banner never went away.** `urgency=critical` never
  auto-expires (freedesktop spec; Plasma ignores `--expire-time` for it), so
  `notify-send --wait` never returned and the daemon killed it — which does
  not retract a banner the server already accepted, it only drops the action
  buttons with their sender. The result was a permanent banner claiming to be
  recording, with a dead Continue button. The banner's lifetime is now managed
  explicitly via `CloseNotification` and ends with the window it describes,
  whether the user answers, the countdown elapses, or the user stops recording
  first. Urgency stays `critical` so the prompt still reaches users in Do Not
  Disturb / presentation mode.
- **Utterances never ended on their own in a noisy room.** A fixed RMS
  threshold cannot separate speech from silence when the input gain is not
  fixed either, and the daemon forces the mic to 91% on activation. Ambient
  noise measured several times `energy_threshold`, so every block scored as
  voiced, no silence gap was ever found, and every utterance was chopped
  mid-word at `--max-utterance-s` — giving Whisper noise-only fragments to
  hallucinate over.
- The session-limit prompt no longer discards a Continue click that lands in
  the last moment of the countdown, and no longer stops a *new* recording
  when the user restarts dictation while the previous prompt is still
  resolving.

### Added

- **`install.py` is version-aware and now actually updates a packaged
  install.** It reports the installed version and this tree's version, exits
  immediately when they match, and otherwise prompts before doing anything
  else. On a packaged system it rebuilds the package from the tree and
  installs it, because configurator mode deliberately never writes Python
  code — previously the installer would wire up the KDE bits, restart the
  daemon on whatever the package already contained, and report success while
  the new code was never deployed. The installed version is read from
  `pacman -Q` rather than from the installed module, since the 0.12.0 package
  shipped a wheel whose `APP_VERSION` still read 0.11.1. The packaged step
  counter also no longer stops at 8/11.
- **`--noise-floor-margin`** — the speech gate now adapts to measured ambient
  noise instead of relying on a fixed threshold. The effective gate is
  `max(--energy-threshold, noise_floor * margin)`, clamped to 8x
  `--energy-threshold` so a loud room cannot push it above the user's own
  voice. `0` restores pure fixed-threshold behavior. The `recording ended:`
  log line now reports the measured `noise_floor` and the `gate` range, and a
  session that heard nothing says which knob to turn.

### Changed

- `--energy-threshold` stays at 700 and keeps its v0.13.0 meaning for
  weak/quiet microphones. It is now the *lower bound* of the adaptive gate
  (and scales its ceiling) rather than the whole gate.

## 0.13.0 — 2026-06-04

### Changed

- **Lower the default speech-detection threshold** (`energy_threshold`
  `1000 → 700`). Weak/quiet digital microphones produce speech below the old
  floor and weren't detected; a normal mic's ambient level sits far above
  both values, so this is safe there. Still tunable with `--energy-threshold`
  (lower for quiet mics, raise for noisy rooms).
- **`install.py` is now packaging-aware.** On a system where the kdictate
  package is installed, the installer runs in *configurator* mode: it
  downloads the model, wires the per-user KDE bits, and enables the
  package's **system** service — skipping the redundant venv and the
  per-user systemd/D-Bus/IBus units that previously shadowed the package
  (and clearing any stale ones from an earlier install). Source checkouts
  keep the venv flow unchanged.
- **No runtime backend fallback — the installer chooses, exactly one.** The
  backend is fixed at install time: `gpu` (or fail) or `cpu`, never both.
  `--backend auto` is removed. A GPU install only provisions the GGML model,
  so the old silent GPU→CPU fallback had no CPU model to load anyway (it would
  error or trigger a surprise 780 MB download mid-dictation). The installer
  now downloads one model and pins the choice (packaged: a systemd drop-in;
  source: the rendered unit).

## 0.12.0 — 2026-06-03

### Added

- **kdictate now ships as a self-contained Arch package** (`packaging/PKGBUILD`)
  that vendors its entire runtime, removing the `whisper.cpp-vulkan` /
  `llama.cpp-vulkan` AUR dependency and the daily-rebuild churn that came with
  it. The GPU transcription stack — a pinned Vulkan `whisper.cpp` with `ggml`
  bundled in — is built into `/usr/lib/kdictate` and version-locked to the
  release, so it changes only when kdictate does. The Python dependencies are
  pip-vendored alongside it (PyGObject excepted — it ships via system
  `python-gobject`). whisper is compiled portable (multi-variant CPU dispatch +
  Vulkan), so one package runs across Ryzen Zen 2 → Zen 5.
- MIT `LICENSE` file.

### Notes

- Not yet turn-key: the speech model is still downloaded on first run, and the
  packaged (system) install and `install.py` (per-user) paths don't yet
  reconcile. See `docs/packaging.md` for the remaining follow-ups.

## 0.11.1 — 2026-05-17

### Fixed

- **IBus engine silently drifting back to `xkb:us::eng` mid-session left
  the daemon healthy but dictation dead.** Symptom: mic activates,
  daemon records and emits `FinalTranscript`, but no text appears in the
  focused field because no KDictate engine instance is subscribed (only
  the layout engine is active). Observed on KDE Plasma Wayland after
  IBus daemon restarts and input-method config reloads — the engine
  remained preloaded but no longer the *active* one, and the v0.11.0
  hot-start fix only covered the at-install case. The daemon now calls
  `ibus engine io.github.pizzimenti.KDictate1` at the start of every
  session (next to `set_default_source_volume`) and logs a single
  warning when it had to heal. The `ibus engine <name>` set is observed
  to occasionally exit non-zero on KDE/Wayland even when the switch
  succeeds, so the helper re-reads the active engine to verify rather
  than trusting the exit code. Best-effort: if `ibus` is missing or
  unresponsive within 2s, the daemon still proceeds to record.

## 0.11.0 — 2026-05-10

### Fixed

- **IBus hot-start broken on fresh KDE Plasma Wayland install.** The
  previous `gdbus VirtualKeyboard.enabled` toggle was a no-op when run
  immediately after install: KWin had a null `InputMethod` in memory
  because `kwriteconfig6` writes to disk but KWin does not re-read
  `kwinrc` during a live session. Replaced with a three-step sequence
  via `qdbus6` (fallback `qdbus`): `org.kde.KWin /KWin reconfigure`
  (makes KWin pick up the new `InputMethod` key), `pkill -x ibus-daemon`
  (clears any stale daemon that would no-op the next signal), and an
  `org.kde.kwin.VirtualKeyboard.enabled` false→true toggle via
  `Properties.Set` (causes KWin to launch the `InputMethod` desktop
  file, which spawns `ibus-ui-gtk3 --enable-wayland-im` with the
  correct Wayland socket, setting `VirtualKeyboard.available=true`).
  `ibus restart` does not work for this — it re-execs the daemon in
  place with the same args, so KWin sees no D-Bus name change and the
  Wayland IM bridge is never launched. Skips the hot-start cleanly
  when `qdbus6`/`qdbus` or `pkill` are unavailable, or when
  `reconfigure` returns non-zero, instead of leaving the user with a
  broken IM stack and no recovery; if either toggle call fails after
  the daemon kill, falls back to `ibus-daemon -r -d` so at least basic
  IBus is running until next login.
- **Installer could hang indefinitely on a healthy model cache.**
  `snapshot_download` ran unconditionally, issuing HEAD/etag requests
  for every file even when nothing needed downloading. A single hung
  request to huggingface.co (silent connection drop) blocked the
  install at step 3 with no progress and no recovery. Added a pre-check
  that uses per-file minimum sizes (model.bin ≥ 1.5 GB, tokenizer.json
  ≥ 2 MB, vocabulary.json ≥ 500 kB, configs ≥ 100 B) — skips the
  network call when the cache is healthy and re-downloads when a file
  is truncated. Same per-file size protection added to the GGML GPU
  model path. To force a re-download, delete the model directory.
- **`troubleshoot.py` false negative on `InputMethod[$e]=` form.** KDE
  KConfig may write the key as `InputMethod=` or `InputMethod[$e]=` (the
  latter signals env-var expansion at read time). The diagnostic check
  now matches both forms; previously the `[$e]=` variant was reported as
  a config error on systems where it was actually correct.
- **`troubleshoot.py` false-green when `pactl` cannot reach the audio
  server.** The audio-device check now gates on `pactl get-default-source`'s
  return code; previously it only inspected stdout and reported PASS
  when stdout was empty because `pactl` failed to contact PulseAudio/
  PipeWire.

### Added

- **`troubleshoot.py`**: diagnostic script that checks every layer of
  the stack (system binaries, installed files, `environment.d` contents,
  `kwinrc` settings, live KWin D-Bus state, IBus processes and engine
  registration, systemd service, D-Bus ping, audio input device) and
  prints a one-liner recovery command (derived from the same `qdbus6`/
  `qdbus` detection logic the installer uses) if anything is
  misconfigured.
- **`docs/architecture-ibus.md`**: documents the KDE Plasma Wayland
  IBus startup lifecycle, the compositor socket restriction that
  prevents manual `ibus-ui-gtk3 --enable-wayland-im` invocation, what
  does NOT work for hot-start (and why), and the working sequence with
  rationale.
- **VAD per-session config dump and end-of-session summary**: at
  recording start the daemon now logs `vad config:
  energy_threshold=… start_speech_blocks=… min_speech_blocks=…
  silence_blocks=… sample_rate=… block_ms=…`, and at recording end
  logs a one-line summary `recording ended: …s, … blocks, … voiced
  (>=…), … committed, peak_rms=…, peak_below_thresh=…`. Makes silent
  recordings diagnosable from the log alone — peak_rms vs.
  energy_threshold tells you whether the mic gain is the issue,
  voiced_blocks vs. commits tells you whether VAD heuristics rejected
  real speech.

### Changed

- **VAD defaults loosened** to accept conversational-volume speech.
  The previous values (raised in 0.8.x to fight Whisper hallucinations
  on ambient noise) were rejecting natural utterances on real
  mic+voice combos; the unconditional `HALLUCINATION_PHRASES` filter
  added in 0.9.x makes the over-tight gates redundant. New defaults:

  | Setting | Before | After |
  |---|---|---|
  | `energy_threshold` | 1500 | 1000 |
  | `silence_ms` | 300 | 600 |
  | `min_speech_ms` | 180 | 120 |
  | `start_speech_ms` | 150 | 90 |
  | decode-skip gate | < 0.5s | < 0.3s |

  `VADConfig` in `audio_common.py` and `daemon_arg_defaults()` in
  `daemon_profiles.py` are now in sync so direct `VADConfig()`
  instantiations don't pick up stale defaults.

## 0.10.1 — 2026-05-04

### Added

- **Session watchdog with continue prompt.** After 30 seconds of
  continuous recording the daemon fires a critical-urgency desktop
  notification with a Continue action button. Clicking Continue
  re-arms the watchdog for another 30 seconds. Letting the
  notification time out (10s default) auto-stops dictation through
  the normal toggle-off path so the accumulated transcript is still
  committed. This guards against the failure mode where dictation is
  left on for hours, accumulating tens of thousands of characters in
  one `FinalTranscript` D-Bus signal — that single oversized commit
  is what wedged IBus' input context for an entire session recently
  (a 7h46m run from 14:16 to 22:02 produced a 37,810-char final
  transcript that broke the input context for the next day).
  New CLI flags `--session-max-recording-s` (default 30) and
  `--session-confirm-timeout-s` (default 10) tune both windows;
  setting `--session-max-recording-s=0` disables the watchdog.

## 0.10.0 — 2026-04-16

### Fixed

- **Restore mic input gain on every activation.** The VAD's
  `energy_threshold` (1500 RMS) assumes the mic is audible, but Plasma
  controls, call apps, and per-app auto-gain can silently drop the
  default source's volume below that floor — producing sessions that
  record cleanly but emit `no speech detected` because the RMS never
  crosses threshold. The daemon now calls
  `pactl set-source-volume @DEFAULT_SOURCE@ 91%` on every start, so
  the next capture has a known-good gain. The pactl call is sandwiched
  between two `_cancel_start` gates: it only runs after mic validation
  passes, and cancellation is re-checked immediately after pactl
  returns (the call can take up to its 3-second timeout), so a stop
  request during startup never mutates system volume or spawns worker
  threads for a session that is about to abort. pactl failures are
  logged but non-fatal — recording still proceeds.

## 0.9.2 — 2026-04-16

### Fixed

- **Drop the RMS gate on hallucination suppression.** The first two
  drops were conditional on the utterance's RMS falling below a low-
  energy ceiling, on the theory that known hallucination phrases only
  appear during near-silence. Real-world testing showed ambient mic
  noise (HVAC, fans, keyboards) produces RMS well above any useful
  ceiling, so the gate leaked hallucinations through. Filtering is now
  unconditional whenever the phrase matches.
- **Fix 3 pre-existing test failures** surfaced during PR #9 review.
  Updated `test_install` regex to match current error messages; updated
  `test_daemon` to patch the functions `main()` actually calls after
  the GPU backend refactor.

## 0.9.1 — 2026-04-16

### Fixed

- **Review-round fixes for the hallucination filter.** Addressed
  feedback from CodeRabbit, Codex, and a manual code-review pass:
  tighter phrase matching, improved normalization of punctuation and
  whitespace before comparison, and clearer logging when a transcript
  is suppressed so users understand why their dictation produced no
  output.

## 0.9.0 — 2026-04-16

### Added

- **Suppress Whisper hallucination phrases.** Whisper models hallucinate
  short phrases like "Thank you", "you", "Bye", and "Okay" when the
  microphone captures ambient noise but no speech. A post-transcription
  filter now suppresses known hallucination phrases when they are the
  entire transcript output (PR #9).

## 0.8.2 — 2026-04-12

### Fixed

- **Chromium/Wayland text insertion regression.** Dictated text was not
  inserted into Chrome text fields; the preedit status animation
  ("Transcribing...") was left in the field instead.
  - Swap commit/clear ordering so `commit_text` arrives before the preedit
    clear. On the Wayland text-input-v3 path each IBus call becomes a
    separate protocol batch; clearing first caused Chrome to finalize the
    animation text.
  - Stop discarding deferred text when the daemon transitions to idle.
    Chrome on Wayland sends spurious focus-out events during transcription;
    the final transcript was deferred correctly but then thrown away before
    focus returned.
  - Always commit deferred text on focus return regardless of daemon state.
  - Remove redundant `hide_preedit_text()` call from the render adapter;
    `update_preedit_text_with_mode(visible=False)` is sufficient and avoids
    an extra signal that Chrome could misinterpret.
