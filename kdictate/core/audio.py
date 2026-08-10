"""Audio-device helpers for the dictation core."""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Sequence
from typing import Final

DEFAULT_PACTL_TIMEOUT_S: Final[float] = 3.0

# pactl re-resolves this on every invocation, so a read and a write that both
# use it can land on two different devices if the default changes in between.
# Callers that must act on one device resolve it to a concrete name first.
DEFAULT_SOURCE_TOKEN: Final[str] = "@DEFAULT_SOURCE@"

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


def set_default_source_volume(
    percent: int | Sequence[int] = MIN_MIC_VOLUME_PERCENT,
    source: str = DEFAULT_SOURCE_TOKEN,
) -> bool:
    """Set ``source``'s volume, either uniformly or per channel.

    ``pactl set-source-volume SOURCE VOLUME [VOLUME ...]`` accepts one value
    per channel, so a sequence sets each channel independently. Passing a
    single value to a multi-channel source sets every channel to it, which
    flattens any balance the user had — see
    :func:`ensure_default_source_volume` for why that matters.

    Returns ``True`` on success, ``False`` if pactl was unavailable or exited
    with a non-zero status. Logs warnings but does not raise — recording should
    still proceed even if the volume adjustment fails.
    """

    percents = [percent] if isinstance(percent, int) else list(percent)
    if not percents:
        return False
    rendered = [f"{value}%" for value in percents]

    try:
        result = _run_pactl("set-source-volume", source, *rendered)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("pactl set-source-volume failed: %s", exc)
        return False

    if result.returncode != 0:
        _LOGGER.warning(
            "pactl set-source-volume %s %s exited %d: %s",
            source, " ".join(rendered), result.returncode, result.stderr.strip(),
        )
        return False
    return True


def read_default_source_name() -> str | None:
    """Return the concrete name of the default source, or ``None``."""

    try:
        result = _run_pactl("get-default-source")
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("pactl get-default-source failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def read_default_source_volumes(
    source: str = DEFAULT_SOURCE_TOKEN,
) -> list[int] | None:
    """Return one percentage per channel for ``source``, or ``None``.

    ``pactl get-source-volume`` prints an entry per channel::

        Volume: front-left: 59637 /  91% / -2.46 dB,   front-right: ... / 91% ...
                balance 0.00

    Every channel is returned rather than just the first. Reading one value
    and writing one value back would set *all* channels to it, so a source
    balanced 40%/90% would be read as 40, "raised" to the floor, and silently
    flattened to 50/50 — lowering a channel, under a contract that promises
    only ever to raise. The trailing ``balance`` line carries no percentage,
    so it does not pollute the match.
    """

    try:
        result = _run_pactl("get-source-volume", source)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("pactl get-source-volume failed: %s", exc)
        return None

    if result.returncode != 0:
        _LOGGER.warning(
            "pactl get-source-volume exited %d: %s",
            result.returncode, result.stderr.strip(),
        )
        return None

    percents = [int(value) for value in re.findall(r"/\s*(\d+)%", result.stdout)]
    if not percents:
        _LOGGER.warning(
            "could not parse a percentage from pactl volume output: %r",
            result.stdout.strip()[:200],
        )
        return None
    return percents


def ensure_default_source_volume(
    minimum_percent: int = MIN_MIC_VOLUME_PERCENT,
) -> bool:
    """Raise capture volume to ``minimum_percent`` only if it is below it.

    Returns ``True`` when the level is acceptable afterwards (either it
    already was, or it was raised successfully), ``False`` when pactl could
    not be consulted or the change failed. Only ever raises, and never touches
    the device at all when ``minimum_percent`` is 0 or less.

    The asymmetry is deliberate. Raising rescues the failure this feature
    exists for — a source that has drifted so low nothing is transcribable —
    while lowering, or pinning to a fixed target, would override a level the
    user chose deliberately and, at a high enough setting, clip the capture
    into uselessness. See :data:`MIN_MIC_VOLUME_PERCENT`.

    Channels are handled individually. A source balanced 40%/90% has only one
    channel below the floor, and writing a single value back would set both to
    it — lowering the 90% channel and flattening the balance, under a contract
    that promises only ever to raise. Each channel is therefore compared and
    written as ``max(channel, minimum_percent)``.

    Concurrency, stated honestly: the check and the change are two separate
    pactl invocations, because PulseAudio/PipeWire expose no conditional
    "raise to at least N" operation to do it in one. The default source is
    resolved to a concrete name first, so the two calls cannot land on
    different devices — but if something else raises *this* device's volume in
    the window between them, this call can still put it back down to the
    floor. The exposure is a few milliseconds once per activation, the result
    is one session at the floor rather than at the user's level, and the next
    activation re-reads and leaves it alone. Closing it properly would mean
    taking a native PipeWire dependency for a race whose worst outcome is a
    single quiet session, which is not a trade worth making here.
    """

    if minimum_percent <= 0:
        return True

    # Resolve once. Passing @DEFAULT_SOURCE@ to both calls would let a
    # default-device switch between them read one device and write another.
    source = read_default_source_name() or DEFAULT_SOURCE_TOKEN

    current = read_default_source_volumes(source)
    if not current:
        # Unknown level: do nothing rather than guess. Pinning blind is what
        # this function was changed to stop doing.
        return False

    if min(current) >= minimum_percent:
        _LOGGER.info(
            "capture volume %s (>= %d%%), left alone",
            "/".join(f"{value}%" for value in current), minimum_percent,
        )
        return True

    targets = [max(value, minimum_percent) for value in current]
    _LOGGER.warning(
        "capture volume %s is below the %d%% floor; raising to %s. Speech "
        "below this level is not reliably detected.",
        "/".join(f"{value}%" for value in current), minimum_percent,
        "/".join(f"{value}%" for value in targets),
    )
    return set_default_source_volume(targets, source)


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
