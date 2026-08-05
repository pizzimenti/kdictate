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
        # Restored from the 700.0 that v0.13.0 set. Lowering the absolute
        # floor was the wrong direction: the ambient noise floor at the mic
        # gain this daemon forces already sits above it, so dropping it only
        # widened the band of room noise that scores as speech. The adaptive
        # noise_floor_margin is what actually separates the two now.
        "energy_threshold": 1000.0,
        "noise_floor_margin": 1.6,
        "silence_ms": 600,
        "min_speech_ms": 120,
        "start_speech_ms": 90,
        "max_utterance_s": 10.0,
        "session_max_recording_s": 30.0,
        "session_confirm_timeout_s": 10.0,
    }
