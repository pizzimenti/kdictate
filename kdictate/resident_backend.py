"""Keep one native STT model loaded, with bounded I/O over private pipes."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import select
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any

from kdictate.audio_common import postprocess_transcript
from kdictate.exceptions import TranscriptionError

logger = logging.getLogger("kdictate.daemon.backend")
MAX_PCM_BYTES = 16000 * 2 * 60
MAX_REPLY_BYTES = 1024 * 1024


class ResidentBackend:
    """Own a native worker and its model for the daemon's lifetime.

    A failed request tears down its worker; the next request may reload the
    same selected engine. No request is replayed and no model fallback occurs.
    """

    def __init__(self, binary: str, model_path: str | Path, *, engine: str = "whisper",
                 n_threads: int = 6, language: str = "en", beam_size: int = 3,
                 no_speech_threshold: float = 0.6, timeout: float = 30.0,
                 startup_timeout: float = 90.0, cpu: bool = False) -> None:
        if engine not in ("whisper", "parakeet"):
            raise ValueError(f"unknown native STT engine: {engine}")
        self.binary = binary
        self.model_path = str(model_path)
        self.engine = engine
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self._command = [binary, engine, self.model_path, str(n_threads), language,
                         str(beam_size), str(no_speech_threshold)]
        if cpu:
            self._command.append("cpu")
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._stderr = None
        self._closed = False
        self.startup_seconds = 0.0
        self.start()

    def _io(self, fd: int, *, deadline: float, data: bytes | None = None, size: int = 0) -> bytes:
        result = bytearray()
        offset = 0
        target = len(data) if data is not None else size
        while offset < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("native STT response timed out")
            readable, writable, _ = select.select(
                [] if data is not None else [fd], [fd] if data is not None else [], [], remaining,
            )
            if not readable and not writable:
                raise TimeoutError("native STT response timed out")
            try:
                if data is not None:
                    count = os.write(fd, memoryview(data)[offset:])
                else:
                    chunk = os.read(fd, min(target - offset, 65536))
                    if not chunk:
                        raise EOFError("native STT worker exited")
                    count = len(chunk)
                    result.extend(chunk)
            except BlockingIOError:
                continue
            if count == 0:
                raise EOFError("native STT pipe closed")
            offset += count
        return bytes(result)

    def _read_reply(self, deadline: float) -> str:
        fd = self._process.stdout.fileno()
        size, = struct.unpack("<I", self._io(fd, deadline=deadline, size=4))
        if size > MAX_REPLY_BYTES:
            raise ValueError("invalid native STT response size")
        return self._io(fd, deadline=deadline, size=size).decode("utf-8")

    def _diagnostic(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, self._stderr.tell() - 4000))
        return self._stderr.read().decode("utf-8", errors="replace").strip()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise TranscriptionError("native STT backend is closed")
            if self._process is not None and self._process.poll() is None:
                return
            self._stop()
            started = time.monotonic()
            try:
                self._stderr = tempfile.TemporaryFile()
                self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
                os.set_blocking(self._process.stdin.fileno(), False)
                os.set_blocking(self._process.stdout.fileno(), False)
                ready = self._read_reply(time.monotonic() + self.startup_timeout)
                if ready != f"READY 1 {self.engine}":
                    raise ValueError(f"unexpected native STT handshake: {ready!r}")
            except (OSError, EOFError, TimeoutError, ValueError) as exc:
                diagnostic = self._diagnostic()
                self._stop()
                raise TranscriptionError(f"{self.engine} initialization failed: {exc}; {diagnostic}") from exc
            self.startup_seconds = time.monotonic() - started
            logger.info("resident %s model loaded in %.2fs (pid %d)",
                        self.engine, self.startup_seconds, self._process.pid)

    def transcribe(self, pcm_chunks: list[Any], audio_seconds: float) -> str:
        if not pcm_chunks:
            return ""
        import numpy as np
        pcm = np.concatenate(pcm_chunks).astype("<i2", copy=False).tobytes()
        if not pcm:
            return ""
        if len(pcm) > MAX_PCM_BYTES:
            raise TranscriptionError("native STT audio exceeds the 60 second request limit")
        with self._lock:
            self.start()
            deadline = time.monotonic() + self.timeout
            try:
                self._io(self._process.stdin.fileno(), deadline=deadline, data=struct.pack("<I", len(pcm)) + pcm)
                text = self._read_reply(deadline)
            except (OSError, EOFError, TimeoutError, ValueError) as exc:
                diagnostic = self._diagnostic()
                self._stop()
                raise TranscriptionError(f"{self.engine} transcription failed: {exc}; {diagnostic}") from exc
            return postprocess_transcript(text.strip())

    def _stop(self) -> None:
        proc, self._process = self._process, None
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            finally:
                if proc.stdout is not None:
                    proc.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop()
