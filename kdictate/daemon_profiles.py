"""Default daemon arguments."""

from __future__ import annotations

from kdictate.app_metadata import DEFAULT_MODEL_DIR
from kdictate.runtime_profile import recommended_shortform_cpu_threads


def daemon_arg_defaults() -> dict[str, object]:
    """Return the daemon argument defaults."""

    return {
        "model_dir": str(DEFAULT_MODEL_DIR),
        "language": "en",
        "sample_rate": 16000,
        "beam_size": 1,
        "condition_on_previous_text": False,
        "vad_filter": True,
        "no_speech_threshold": 0.6,
        "cpu_threads": recommended_shortform_cpu_threads(),
        "compute_type": "int8",
        "block_ms": 30,
        # Kept at the 700.0 v0.13.0 chose for weak/quiet digital microphones,
        # whose speech falls below the older 1000. Raising it back would lock
        # those microphones out again, and it is no longer the knob that
        # handles noisy rooms -- noise_floor_margin is.
        "energy_threshold": 700.0,
        "noise_floor_margin": 1.6,
        "silence_ms": 600,
        "min_speech_ms": 120,
        "start_speech_ms": 90,
        "max_utterance_s": 10.0,
        "session_max_recording_s": 30.0,
        "session_confirm_timeout_s": 10.0,
    }
