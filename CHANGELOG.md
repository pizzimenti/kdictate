# Changelog

## 0.17.0

Root-causes and fixes the "every install breaks dictation until reboot"
pattern that ran through 0.14–0.16, and adds the observability that would
have made it a one-round diagnosis.

### The root cause

`refresh_ibus_registry` killed the session's ibus-daemon on every install,
then toggled KWin's `org.kde.kwin.VirtualKeyboard.enabled` D-Bus property to
make KWin respawn the input-method stack. Plasma 6.7 removed that property,
so the toggle failed on every install — *after* the daemon was already dead —
and the fallback started a bare `ibus-daemon -r -d` with **no Wayland IM
bridge** (`ibus-ui-gtk3 --enable-wayland-im`). Diagnosed 2026-08-10 on the
0.16.0 install: the daemon recorded/decoded/emitted, the engine received the
transcript and committed it (all logged), yet `ss -xp` showed no compositor
or application attached to the new daemon's socket. Committed text had
nowhere to go. Only a relogin — which lets KWin respawn the bridge — restored
typing, which is why every version "worked after reboot."

### Fixed

- **The installer never kills a healthy ibus-daemon.** Shipping new engine
  code needs only `ibus write-cache` plus clearing old engine processes
  (0.16.0's guard); IBus respawns engines on demand. The daemon kill and the
  defunct property toggle are gone.
- **Missing-bridge repair via KWin.** When the Wayland IM bridge is already
  absent (the broken-session shape older installers left behind), the
  installer now relaunches the stack the supported way: flip the kwinrc
  `[Wayland] InputMethod` entry and ask KWin to reconfigure — the same
  mechanism the Virtual Keyboard KCM uses — then verify the bridge process
  actually appeared.

### Added

- **Persistent install log** at `~/.local/state/kdictate/install.log`:
  version transitions, every IBus process the installer touches (with PIDs),
  bridge health decisions, repair outcomes, and the final self-check verdict.
- **Post-install self-check.** The installer verifies the full chain —
  daemon bus name, engine process, active engine, ibus-daemon, Wayland IM
  bridge — and reports specific problems instead of printing an unconditional
  success banner over a severed stack.
- **Build dependencies are removed after a successful rebuild** (the
  `makepkg --rmdeps` half, completing the hand-rolled `--syncdeps` from
  0.14.1). The removal list is the exact `pacman -Qq` diff of the install
  transaction — dependencies included, pre-existing packages out of reach
  by construction — so in the normal case rebuilds no longer leave
  `python-build`/`python-installer`/`python-wheel` behind as orphans. The
  consent prompt announces the removal up front. Best-effort by design: if
  pacman refuses the removal, the packages stay behind as `--asdeps`
  orphans for the next sweep and `install.log` records why; a failed
  cleanup never fails an otherwise-successful install. The preflight probe
  now also checks `python-wheel`, aligning it with the PKGBUILD's
  makedepends.
- **Engine log identity.** Each engine instance logs under a numbered logger
  (`…engine1`, `…engine2`), the process logs its PID at startup, and commits
  log the enabled/focused/daemon-state gate they passed — so "committed but
  nothing typed" is attributable to the correct instance and provably beyond
  kdictate.

The daemon runtime is otherwise unchanged from 0.16.0 (= 0.13.0 behaviour).

## 0.16.0

Rolls the daemon back to its 0.13.0 behaviour. Everything in 0.14.0–0.14.2
is reverted; 0.15.0 was abandoned unreleased. Two pieces of tooling are
carried forward, and nothing else.

### Reverted

- **All of 0.14.x.** The IBus engine-activation work that motivated it was
  chasing a problem that does not exist here. Over 30 days of daemon logs the
  v0.11.1 self-heal recorded **81 failures and 0 successful heals** — the
  probe was blind, so every session logged an error while the engine was
  already active the whole time. The 0.14.0 "fix" restored the probe and the
  errors stopped, but the heal it enables has still never once needed to
  fire. That is the whole of what 0.14.0's headline change achieved: it
  silenced a check complaining about itself.
- Also reverted with it: the adaptive noise-floor VAD gate (off by default,
  so inert), the notification-lifetime rework, the mic-gain change, and the
  session-limit prompt changes. None had evidence of fixing a problem the
  user was actually experiencing, and between them they caused three
  user-visible regressions.

### Kept

- **Stale IBus engine processes are cleared on upgrade.** The one change from
  this line of work with direct evidence behind it. An engine reads its
  script once at spawn and keeps running it, so replacing the package leaves
  the live engine on the old code while IBus spawns a second copy; both
  receive the daemon's `FinalTranscript` broadcast but only one can hold the
  focused input context, and dictation silently produces nothing until the
  next reboot. `refresh_ibus_registry` killed `ibus-daemon` with `pkill -x`,
  an exact *name* match that never reached `python …/ibus-engine-kdictate` —
  so killing the daemon orphaned the engines instead of reaping them, and
  they survived `ibus restart` too. Confirmed on a live system: clearing the
  orphan restored dictation immediately.
- **`install.py` rebuilds and installs the package**, quietly, with its build
  dependencies handled rather than punted. Missing dependencies are detected
  during preflight — before the checklist, so the pacman transaction and
  password prompt happen in the open — named, and installed with consent
  (`--asdeps`, leaving removal policy to the user's own orphan sweep). The
  probes run under `-I` so neither the working directory nor `PYTHONPATH` can
  shadow the module being checked. `--reconfigure` re-runs the configuration
  steps when the version already matches, and repairs a Ctrl+Space binding
  that Plasma has reset rather than assuming the section's presence means it
  is intact.
- A bare `pytest` at the repo root works again; it previously recursed into
  makepkg's staging trees and reported 142 collection errors.

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
