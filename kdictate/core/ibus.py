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
import os
import subprocess
from typing import Final

from kdictate.constants import DBUS_INTERFACE

# 2s caps the worst case latency added to a session-start. ``ibus engine``
# on a healthy desktop returns in <50ms; the 2s budget exists to bound a
# wedged ``ibus-daemon`` rather than to allow a slow happy path.
DEFAULT_IBUS_TIMEOUT_S: Final[float] = 2.0

# The ``ibus`` CLI locates its bus socket by display name, not by D-Bus:
# ``~/.config/ibus/bus/<machine-id>-unix-wayland-0`` on Wayland versus
# ``…-unix-0`` on X11. With WAYLAND_DISPLAY unset it silently falls back to
# the X11 name and reads a stale socket file left over from an earlier
# session, so every invocation dies with::
#
#     ibus_bus_get_global_engine: assertion 'IBUS_IS_BUS (bus)' failed
#     No engine is set.
#
# which this module then reports as "failed to switch active IBus engine".
# DBUS_SESSION_BUS_ADDRESS being correct is *not* sufficient — the socket
# path is chosen before D-Bus is ever consulted.
#
# The daemon lands in that state whenever it starts before the session
# manager exports the display variables into the systemd user environment.
# The unit now orders itself after graphical-session.target so the common
# case inherits a usable environment, but a user service that was enabled
# by hand (or a daemon that outlives an ibus-daemon restart) can still be
# holding an env with no display in it — so resolve the variables at call
# time instead of trusting whatever we happened to be spawned with.
#
# WAYLAND_DISPLAY leads because on a Wayland session DISPLAY alone does not
# help: ibus still computes the wayland socket name and ignores it.
_DISPLAY_ENV_KEYS: Final[tuple[str, ...]] = ("WAYLAND_DISPLAY", "DISPLAY")

# Aliased to DBUS_INTERFACE because the IBus engine name and the daemon's
# D-Bus interface name are the same string by construction (see
# ibus_engine/engine.py: ``ENGINE_NAME = DBUS_INTERFACE``). Re-export under
# a self-describing name so callers reading this module do not need to know
# that historical coupling.
KDICTATE_ENGINE_NAME: Final[str] = DBUS_INTERFACE

_LOGGER = logging.getLogger(__name__)

# Resolved once per process: the display name cannot change under a running
# session, and re-querying systemd on every session-start would put a second
# subprocess on the hot path. ``ensure_active_engine`` drops the cache when a
# query fails so a daemon that started against a since-replaced session can
# still recover without a restart.
_display_env_cache: dict[str, str] | None = None


def _display_env_from_systemd() -> dict[str, str]:
    """Read the display variables back from the systemd user manager.

    ``systemctl --user show-environment`` is the authoritative record of what
    the session manager exported once the graphical session came up, which is
    exactly the state a too-early daemon start missed. Best-effort like every
    other helper here: an empty dict just means we fall through to whatever
    the process environment already had.
    """

    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=DEFAULT_IBUS_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("systemctl --user show-environment failed: %s", exc)
        return {}
    if result.returncode != 0:
        _LOGGER.warning(
            "systemctl --user show-environment exited %d: %s",
            result.returncode, result.stderr.strip(),
        )
        return {}

    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in _DISPLAY_ENV_KEYS:
            # systemd shell-quotes values that need it; display names never
            # do, so stripping a symmetric quote pair is enough here.
            found[key] = value.strip().strip("'\"")
    return found


def _display_env() -> dict[str, str]:
    """Return the display variables to hand the ``ibus`` CLI.

    The process environment wins wherever it is already populated — if we
    were started with a display we are in the session that owns it. systemd
    is consulted only to fill a missing WAYLAND_DISPLAY, which is the one
    variable whose absence actually breaks socket resolution.
    """

    global _display_env_cache
    if _display_env_cache is not None:
        return _display_env_cache

    resolved = {
        key: os.environ[key] for key in _DISPLAY_ENV_KEYS if os.environ.get(key)
    }
    if "WAYLAND_DISPLAY" not in resolved:
        for key, value in _display_env_from_systemd().items():
            resolved.setdefault(key, value)
        if resolved:
            _LOGGER.info("resolved display env for ibus: %s", sorted(resolved))
    _display_env_cache = resolved
    return resolved


def _reset_display_env_cache() -> None:
    """Force the next :func:`_display_env` call to re-resolve."""

    global _display_env_cache
    _display_env_cache = None


def _run_ibus(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the ``ibus`` CLI with UTF-8 decoding and the module timeout.

    UTF-8 + ``errors="replace"`` mirrors ``core.audio._run_pactl`` — engine
    names are ASCII today but the locale of the invoking shell is not
    guaranteed, and a UnicodeDecodeError here would otherwise crash the
    helper instead of degrading to a logged warning.

    The environment is the inherited one plus any display variables we had
    to recover (see :data:`_DISPLAY_ENV_KEYS`); passing a full env rather
    than only the overrides keeps DBUS_SESSION_BUS_ADDRESS and friends
    intact.
    """

    return subprocess.run(
        ["ibus", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=DEFAULT_IBUS_TIMEOUT_S,
        env={**os.environ, **_display_env()},
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
    if current is None:
        # Either ibus-daemon is genuinely unreachable or our cached display
        # env points at a session that has since been replaced. Re-resolving
        # costs one subprocess on an already-failing path and turns the
        # second case from "silently types nothing forever" into a heal.
        _reset_display_env_cache()
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
