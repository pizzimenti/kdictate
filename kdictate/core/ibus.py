"""Daemon-side IBus state helpers.

This module is the small surface the daemon process uses to keep IBus's
active input method in sync with the KDictate engine. It is intentionally
separate from ``kdictate/ibus_engine/`` — that subpackage *implements* the
engine that ``ibus-daemon`` spawns on the IBus side; this module only
inspects and nudges IBus from the daemon side, at session-start.

Why shell out to the ``ibus`` CLI instead of going through the IBus Python
bindings (which the engine itself uses):

  - The CLI is a tiny stable contract (``ibus engine`` / ``ibus engine
    <name>``) that hides every version-specific quirk of IBus's D-Bus
    layout. We do not need the rich engine introspection that the bindings
    provide; we just need to know "is the active engine ours" and "make it
    ours" idempotently.
  - The bindings would otherwise pull a non-trivial dep into the daemon
    process whose only consumer of IBus today is a 30-line state poke.
  - subprocess gives us a hard timeout so a stuck ``ibus-daemon`` cannot
    hang the recording hot-path; using the bindings synchronously would
    block on the GLib main loop with no easy escape.

Failure mode philosophy: every helper here is best-effort. If the ``ibus``
binary is missing, ``ibus-daemon`` is dead, or D-Bus has wedged, the
recording session must still proceed — the user can read transcripts from
the CLI/log and at least diagnose. We never raise into the daemon's
control loop.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Final

from kdictate.constants import DBUS_INTERFACE

# 2s caps the worst case latency added to a session-start. ``ibus engine``
# on a healthy desktop returns in <50ms; the 2s budget exists to bound a
# wedged ``ibus-daemon`` rather than to allow a slow happy path.
DEFAULT_IBUS_TIMEOUT_S: Final[float] = 2.0

# Aliased to DBUS_INTERFACE because the IBus engine name and the daemon's
# D-Bus interface name are the same string by construction (see
# ibus_engine/engine.py: ``ENGINE_NAME = DBUS_INTERFACE``). Re-export under
# a self-describing name so callers reading this module do not need to know
# that historical coupling.
KDICTATE_ENGINE_NAME: Final[str] = DBUS_INTERFACE

_LOGGER = logging.getLogger(__name__)


def _run_ibus(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the ``ibus`` CLI with UTF-8 decoding and the module timeout.

    UTF-8 + ``errors="replace"`` mirrors ``core.audio._run_pactl`` — engine
    names are ASCII today but the locale of the invoking shell is not
    guaranteed, and a UnicodeDecodeError here would otherwise crash the
    helper instead of degrading to a logged warning.
    """

    return subprocess.run(
        ["ibus", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=DEFAULT_IBUS_TIMEOUT_S,
    )


def _read_active_engine() -> str | None:
    """Return the active IBus engine name, or ``None`` if it cannot be read.

    Swallowed failure modes:

      - ``ibus`` binary not on PATH (OSError / FileNotFoundError).
      - ``ibus-daemon`` not running (the CLI exits non-zero and writes a
        D-Bus error to stderr).
      - Timeout (``ibus-daemon`` hung on a D-Bus call).

    Each is logged at WARNING so a session that records but does not type
    leaves a breadcrumb in the daemon log, while the caller still gets a
    clean ``None`` to drive its own fallback.
    """

    try:
        result = _run_ibus("engine")
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("ibus engine query failed: %s", exc)
        return None
    if result.returncode != 0:
        _LOGGER.warning(
            "ibus engine query exited %d: %s",
            result.returncode, result.stderr.strip(),
        )
        return None
    return result.stdout.strip() or None


def ensure_active_engine(engine_name: str = KDICTATE_ENGINE_NAME) -> bool:
    """Make sure ``engine_name`` is the active IBus engine.

    Returns ``True`` if the engine matches (either already, or after a
    successful heal). Returns ``False`` if ``ibus`` is unavailable or the
    switch did not stick.

    Why this exists
    ---------------
    On KDE/Wayland the active IBus engine can silently revert to the
    keyboard layout (e.g. ``xkb:us::eng``) after IBus daemon restarts or
    input-method config reloads. The KDictate engine remains *preloaded*
    (in ``preload-engines``) but is no longer the *active* one. When that
    happens, the kdictate daemon still records audio and emits
    ``FinalTranscript`` over D-Bus, but no KDictate engine instance is
    subscribed to those signals, so nothing types into the focused field.
    Calling this at the start of every session turns that failure mode
    into a self-healing one — at the cost of a single ``ibus engine``
    query on the hot path and three calls on the cold (heal) path.

    Hot-path cost
    -------------
    Fast path (engine already active): one ``ibus engine`` invocation,
    ~10–50ms on a healthy system. This is the overwhelmingly common case
    once a session is established and the user has not touched IBus
    config, so the per-session tax is small.

    Cold path (heal needed): three calls (query → set → verify),
    ~30–150ms total. Acceptable because the alternative is a recording
    that produces no output.

    Why verify after set
    --------------------
    ``ibus engine <name>`` is observed to occasionally exit non-zero on
    KDE/Wayland *even when the switch succeeds* (likely a races between
    the CLI's NameOwnerChanged listener and the GlobalEngineChanged
    signal). Trusting the exit code would produce spurious "switch
    failed" warnings on what was actually a successful heal. Re-reading
    the active engine gives us ground truth.

    Logging policy
    --------------
    No log on the fast path — this runs at session frequency and silent
    success is the right behavior. A WARNING is emitted only when a heal
    actually occurred, so the frequency of those warnings in the daemon
    log is the diagnostic for "how often is IBus drifting under us."
    """

    current = _read_active_engine()
    if current == engine_name:
        return True

    try:
        _run_ibus("engine", engine_name)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("ibus engine %s failed: %s", engine_name, exc)
        return False

    confirmed = _read_active_engine()
    if confirmed == engine_name:
        _LOGGER.warning(
            "active IBus engine was %r; switched to %r for this session",
            current, engine_name,
        )
        return True

    _LOGGER.warning(
        "failed to switch active IBus engine to %r (still %r)",
        engine_name, confirmed,
    )
    return False
