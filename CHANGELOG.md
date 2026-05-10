# Changelog

## Unreleased

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
  Wayland IM bridge is never launched. If neither `qdbus6` nor `qdbus`
  is on `PATH`, the installer skips the hot-start (no recovery path
  available); the new InputMethod takes effect at the next login.
- **`troubleshoot.py` false negative on `InputMethod[$e]=` form.** KDE
  KConfig may write the key as `InputMethod=` or `InputMethod[$e]=` (the
  latter signals env-var expansion at read time). The diagnostic check
  now matches both forms; previously the `[$e]=` variant was reported as
  a config error on systems where it was actually correct.

### Added

- `troubleshoot.py`: diagnostic script that checks every layer of the
  stack (system binaries, installed files, `environment.d` contents,
  `kwinrc` settings, live KWin D-Bus state, IBus processes and engine
  registration, systemd service, D-Bus ping, audio input device) and
  prints a one-liner fix if anything is misconfigured.
- `docs/architecture-ibus.md`: documents the KDE Plasma Wayland IBus
  startup lifecycle, the compositor socket restriction that prevents
  manual `ibus-ui-gtk3 --enable-wayland-im` invocation, and the working
  hot-start sequence with rationale.

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
