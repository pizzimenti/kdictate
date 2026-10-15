"""Exercise the real dependency boundary that mocks cannot verify."""
import io
import wave

import numpy as np
import pytest


def test_faster_whisper_decodes_pcm_wav_with_supported_pyav():
    # PyAV 19 removed metadata_errors, still passed by faster-whisper 1.2.1.
    # Decode through the actual library to catch incompatible future updates.
    audio = pytest.importorskip("faster_whisper.audio")
    samples = np.arange(-1024, 1024, dtype="<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(samples.tobytes())
    stream.seek(0)
    decoded = audio.decode_audio(stream, sampling_rate=16000)
    np.testing.assert_array_equal(decoded, samples.astype(np.float32) / 32768.0)
