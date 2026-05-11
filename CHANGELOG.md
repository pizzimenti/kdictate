# Changelog

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
