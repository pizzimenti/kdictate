"""Transcription backend abstraction.

Two backends share a common interface so the daemon can swap between
faster-whisper (CPU, default) and whisper.cpp (Vulkan GPU, optional)
without changing the VAD, D-Bus, or IBus layers.

The CPU backend delegates to ``transcribe_pcm`` in ``audio_common``,
keeping a single source of truth for the faster-whisper call.  The GPU
backend shells out to the ``whisper-cli`` binary, piping PCM audio as
an in-memory WAV file via stdin.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Protocol

from kdictate.app_metadata import GGML_MODEL_PATH
from kdictate.audio_common import transcribe_pcm

logger = logging.getLogger("kdictate.daemon.backend")


class TranscriptionBackend(Protocol):
    """Minimal contract every transcription backend must satisfy."""

    def transcribe(self, pcm_chunks: list[Any], audio_seconds: float) -> str:
        """Transcribe int16 PCM chunks and return normalized text."""
        ...  # pragma: no cover


# ------------------------------------------------------------------
# faster-whisper (CPU) backend
# ------------------------------------------------------------------


class FasterWhisperBackend:
    """CPU transcription via faster-whisper / CTranslate2 (int8)."""

    def __init__(self, model: Any, *, language: str, beam_size: int,
                 no_speech_threshold: float, condition_on_previous_text: bool,
                 vad_filter: bool) -> None:
        self.model = model
        self.language = language
        self.beam_size = beam_size
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter

    def transcribe(self, pcm_chunks: list[Any], audio_seconds: float) -> str:
        return transcribe_pcm(
            self.model, pcm_chunks,
            language=self.language, beam_size=self.beam_size,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=self.vad_filter,
        )


# ------------------------------------------------------------------
# whisper.cpp (Vulkan GPU) backend
# ------------------------------------------------------------------

# Optimal defaults determined by benchmarking on a Ryzen 5 8640HS
# with Radeon 760M iGPU.  Q8_0 is 15% faster than FP16 with no
# measurable accuracy loss.  Beam 3 is free (encoder-bound) and
# preserves capitalization/punctuation.  Flash attention gives a
# small but consistent speed win on Vulkan.
_GPU_BEAM_SIZE = 3
_GPU_FLASH_ATTN = True


def _pcm_to_wav_bytes(pcm_chunks: list[Any], sample_rate: int = 16000) -> bytes:
    """Encode int16 PCM chunks as an in-memory WAV file."""
    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.concatenate(pcm_chunks).tobytes())
    return buf.getvalue()


def find_whisper_cpp() -> str | None:
    """Return the path to a whisper.cpp binary, or None."""
    for name in ("whisper-cli", "whisper-cpp", "main"):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


class WhisperCppBackend:
    """GPU transcription via whisper.cpp CLI with Vulkan acceleration."""

    def __init__(self, binary: str, model_path: str | Path, *,
                 language: str = "en", beam_size: int = _GPU_BEAM_SIZE,
                 n_threads: int = 6, flash_attn: bool = _GPU_FLASH_ATTN,
                 ) -> None:
        self.binary = binary
        self.model_path = str(model_path)
        self.language = language
        self.beam_size = beam_size
        self.n_threads = n_threads
        self.flash_attn = flash_attn

    def transcribe(self, pcm_chunks: list[Any], audio_seconds: float) -> str:
        if not pcm_chunks:
            return ""

        cmd = [
            self.binary,
            "--model", self.model_path,
            "--language", self.language,
            "--beam-size", str(self.beam_size),
            "--threads", str(self.n_threads),
            "--no-timestamps", "--no-prints",
            "--output-txt", "--output-file", "-",
            "--file", "-",
        ]
        if self.flash_attn:
            cmd.append("--flash-attn")

        try:
            result = subprocess.run(
                cmd, input=_pcm_to_wav_bytes(pcm_chunks),
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("whisper.cpp timed out after 30s")
            return ""
        except OSError as exc:
            logger.error("whisper.cpp exec failed: %s", exc)
            return ""

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            logger.warning("whisper.cpp exited %d: %s", result.returncode, stderr[:200])
            return ""

        text = result.stdout.decode(errors="replace").strip()
        if not text:
            return ""
        return " ".join(text.replace("\r", " ").replace("\n", " ").split())


# ------------------------------------------------------------------
# Construction helpers
# ------------------------------------------------------------------


def create_cpu_backend(model: Any, config: Any) -> FasterWhisperBackend:
    """Build the faster-whisper CPU backend from a DictationConfig."""
    return FasterWhisperBackend(
        model,
        language=config.language,
        beam_size=config.beam_size,
        no_speech_threshold=config.no_speech_threshold,
        condition_on_previous_text=config.condition_on_previous_text,
        vad_filter=config.vad_filter,
    )


def _probe_whisper_cpp(binary: str, model_path: str) -> bool:
    """Feed 1 s of silence to whisper.cpp to verify it starts correctly.

    Catches Vulkan driver failures, missing libraries, and corrupt model
    files before real dictation starts.
    """
    import numpy as np

    try:
        result = subprocess.run(
            [binary, "--model", model_path, "--language", "en",
             "--no-timestamps", "--no-prints",
             "--output-txt", "--output-file", "-", "--file", "-"],
            input=_pcm_to_wav_bytes([np.zeros(16000, dtype=np.int16)]),
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("whisper.cpp probe failed: %s", exc)
        return False

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        logger.warning("whisper.cpp probe exited %d: %s", result.returncode, stderr[:300])
        return False

    logger.info("whisper.cpp probe succeeded")
    return True


def create_gpu_backend(config: Any) -> WhisperCppBackend | None:
    """Try to build a whisper.cpp GPU backend.  Returns None on failure."""
    binary = find_whisper_cpp()
    if binary is None:
        logger.info("whisper.cpp not found on PATH; GPU backend unavailable")
        return None

    if not GGML_MODEL_PATH.is_file():
        logger.info("GGML model not found at %s; GPU backend unavailable", GGML_MODEL_PATH)
        return None

    logger.info("GPU backend: whisper.cpp=%s model=%s", binary, GGML_MODEL_PATH)

    if not _probe_whisper_cpp(binary, str(GGML_MODEL_PATH)):
        logger.warning("whisper.cpp probe failed; GPU backend unavailable")
        return None

    return WhisperCppBackend(
        binary, GGML_MODEL_PATH,
        language=config.language,
        n_threads=config.cpu_threads,
    )
