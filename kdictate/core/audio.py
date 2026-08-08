"""Audio-device helpers for the dictation core."""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Final

DEFAULT_PACTL_TIMEOUT_S: Final[float] = 3.0

# Floor, not a target. PR #10 pinned the capture volume to 91% on every
# activation because the source had drifted to 40% and speech fell below the
# VAD threshold, so every session logged "no speech detected". Pinning fixed
# that case and created a worse one: on a source that was already healthy it
# drives capture into clipping, and a clipped signal has no dynamic range left
# for an energy VAD to work with. Measured on the author's hardware at 91%:
# peak_rms 32768 (int16 maximum — i.e. clipping), noise floor ~9000 against a
# speaking level of ~11000-13000. A ratio of 1.3 leaves no gap between "room"
# and "voice", which is why every fixed threshold tried (1500, 1000, 700) and
# the adaptive gate all failed to find a silence gap.
#
# So the daemon now only rescues the drift case and otherwise leaves the level
# alone: above this floor the gain is the user's to choose, and theirs to keep.
MIN_MIC_VOLUME_PERCENT: Final[int] = 50

_LOGGER = logging.getLogger(__name__)


def _run_pactl(*args: str) -> subprocess.CompletedProcess[str]:
    """Run pactl with UTF-8 decoding that survives odd locale settings."""

    return subprocess.run(
        ["pactl", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=DEFAULT_PACTL_TIMEOUT_S,
    )


def set_default_source_volume(percent: int = MIN_MIC_VOLUME_PERCENT) -> bool:
    """Set the default PulseAudio/PipeWire source volume to ``percent``.

    Returns ``True`` on success, ``False`` if pactl was unavailable or exited
    with a non-zero status. Logs warnings but does not raise — recording should
    still proceed even if the volume adjustment fails.
    """

    try:
        result = _run_pactl("set-source-volume", "@DEFAULT_SOURCE@", f"{percent}%")
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("pactl set-source-volume failed: %s", exc)
        return False

    if result.returncode != 0:
        _LOGGER.warning(
            "pactl set-source-volume @DEFAULT_SOURCE@ %d%% exited %d: %s",
            percent, result.returncode, result.stderr.strip(),
        )
        return False
    return True


def read_default_source_volume() -> int | None:
    """Return the default source's volume as a percentage, or ``None``.

    ``pactl get-source-volume`` prints one entry per channel::

        Volume: front-left: 59637 /  91% / -2.46 dB,   front-right: ...

    The channels are set together here, so the first percentage is taken as
    representative rather than trying to reconcile a per-channel imbalance the
    daemon did not create and should not silently flatten.
    """

    try:
        result = _run_pactl("get-source-volume", "@DEFAULT_SOURCE@")
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("pactl get-source-volume failed: %s", exc)
        return None

    if result.returncode != 0:
        _LOGGER.warning(
            "pactl get-source-volume exited %d: %s",
            result.returncode, result.stderr.strip(),
        )
        return None

    match = re.search(r"(\d+)%", result.stdout)
    if match is None:
        _LOGGER.warning(
            "could not parse a percentage from pactl volume output: %r",
            result.stdout.strip()[:200],
        )
        return None
    return int(match.group(1))


def ensure_default_source_volume(
    minimum_percent: int = MIN_MIC_VOLUME_PERCENT,
) -> bool:
    """Raise capture volume to ``minimum_percent`` only if it is below it.

    Returns ``True`` when the level is acceptable afterwards (either it
    already was, or it was raised successfully), ``False`` when pactl could
    not be consulted or the change failed. Never lowers the volume, and never
    touches it at all when ``minimum_percent`` is 0 or less.

    The asymmetry is deliberate. Raising rescues the failure this feature
    exists for — a source that has drifted so low nothing is transcribable —
    while lowering, or pinning to a fixed target, would override a level the
    user chose deliberately and, at a high enough setting, clip the capture
    into uselessness. See :data:`MIN_MIC_VOLUME_PERCENT`.
    """

    if minimum_percent <= 0:
        return True

    current = read_default_source_volume()
    if current is None:
        # Unknown level: do nothing rather than guess. Pinning blind is what
        # this function was changed to stop doing.
        return False

    if current >= minimum_percent:
        _LOGGER.info("capture volume %d%% (>= %d%%), left alone",
                     current, minimum_percent)
        return True

    _LOGGER.warning(
        "capture volume %d%% is below the %d%% floor; raising it. Speech below "
        "this level is not reliably detected.",
        current, minimum_percent,
    )
    return set_default_source_volume(minimum_percent)


def resolve_default_input_device() -> tuple[str, bool]:
    """Return ``(description, usable)`` for the default PulseAudio/PipeWire source."""

    try:
        result = _run_pactl("get-default-source")
        source_name = result.stdout.strip()
    except Exception:  # noqa: BLE001
        return ("unknown", False)

    if not source_name:
        return ("none", False)

    # If the default is a monitor (speaker loopback), try to find a real
    # input device before giving up.
    if source_name.endswith(".monitor"):
        fallback = _find_first_real_input()
        if fallback is not None:
            name, description = fallback
            _LOGGER.info(
                "Default source %s is a monitor; switching to %s (%s)",
                source_name, name, description,
            )
            try:
                result = _run_pactl("set-default-source", name)
                if result.returncode != 0:
                    _LOGGER.warning("pactl set-default-source %s exited %d", name, result.returncode)
                    return (source_name, False)
            except Exception:  # noqa: BLE001
                _LOGGER.warning("Failed to set default source to %s", name)
                return (source_name, False)
            return (description, True)
        return (source_name, False)

    return _describe_source(source_name)


def _describe_source(source_name: str) -> tuple[str, bool]:
    """Look up the human-readable description for a named source."""

    try:
        result = _run_pactl("list", "sources")
        in_target = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            parts = stripped.split(None, 1)
            if stripped.startswith("Name:") and len(parts) > 1 and parts[1] == source_name:
                in_target = True
            elif in_target and stripped.startswith("Description:"):
                return (stripped.split(":", 1)[1].strip(), True)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Failed to resolve input device description: %s", exc)

    return (source_name, True)


def _find_first_real_input() -> tuple[str, str] | None:
    """Scan pactl sources for the first non-monitor input device.

    Returns ``(source_name, description)`` or ``None`` if no real input
    device exists.
    """

    try:
        result = _run_pactl("list", "sources", "short")
    except Exception:  # noqa: BLE001
        return None

    candidates: list[str] = []
    for line in result.stdout.strip().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            name = fields[1]
            if not name.endswith(".monitor"):
                candidates.append(name)

    if not candidates:
        return None

    # Return the first real input with its description.
    for name in candidates:
        desc, usable = _describe_source(name)
        if usable:
            return (name, desc)

    return None
