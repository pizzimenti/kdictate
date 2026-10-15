"""Use real private pipes to test worker lifetime and protocol failures."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kdictate.exceptions import TranscriptionError
from kdictate.resident_backend import ResidentBackend


WORKER = '''#!/usr/bin/python
import os, struct, sys, time
engine, mode = sys.argv[1:3]
def send(data):
    sys.stdout.buffer.write(struct.pack("<I", len(data)) + data)
    sys.stdout.buffer.flush()
if mode == "startup-failure":
    print("model rejected", file=sys.stderr)
    sys.exit(1)
send(("READY 1 " + ("wrong" if mode == "bad-handshake" else engine)).encode())
while True:
    header = sys.stdin.buffer.read(4)
    if not header: break
    size, = struct.unpack("<I", header)
    data = sys.stdin.buffer.read(size)
    if mode == "timeout": time.sleep(10)
    if mode == "crash": sys.exit(3)
    if mode == "bad-size":
        sys.stdout.buffer.write(struct.pack("<I", 2**30))
        sys.stdout.buffer.flush()
        continue
    if mode == "invalid-utf8": send(b"\\xff")
    elif mode == "partial":
        sys.stdout.buffer.write(struct.pack("<I", 100) + b"x")
        sys.stdout.buffer.flush()
        sys.exit(0)
    else: send(("hello " + str(len(data))).encode())
'''


class ResidentBackendTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.worker = Path(directory.name) / "worker"
        self.worker.write_text(WORKER)
        self.worker.chmod(0o755)

    def backend(self, mode="normal", **kwargs):
        backend = ResidentBackend(str(self.worker), mode, **kwargs)
        self.addCleanup(backend.close)
        return backend

    def test_model_process_is_reused_across_independent_requests(self):
        backend = self.backend()
        pid = backend._process.pid
        self.assertEqual(backend.transcribe([np.zeros(16000, dtype=np.int16)], 1), "hello 32000")
        self.assertEqual(backend.transcribe([np.zeros(8000, dtype=np.int16)], .5), "hello 16000")
        self.assertEqual(backend._process.pid, pid)
        proc = backend._process
        backend.close()
        self.assertIsNotNone(proc.poll())

    def test_dead_worker_restarts_the_same_selected_engine(self):
        backend = self.backend(engine="parakeet")
        proc = backend._process
        proc.kill()
        proc.wait(timeout=1)
        self.assertEqual(backend.transcribe([np.zeros(100, dtype=np.int16)], .01), "hello 200")
        self.assertNotEqual(backend._process.pid, proc.pid)
        self.assertEqual(backend.engine, "parakeet")

    def test_protocol_failures_reap_the_process(self):
        for mode in ("crash", "bad-size", "partial", "invalid-utf8", "timeout"):
            with self.subTest(mode=mode):
                backend = self.backend(mode, timeout=.05)
                proc = backend._process
                with self.assertRaises(TranscriptionError):
                    backend.transcribe([np.zeros(100, dtype=np.int16)], .01)
                self.assertIsNotNone(proc.poll())
                self.assertIsNone(backend._process)

    def test_startup_failure_and_handshake_mismatch_are_reported(self):
        for mode in ("startup-failure", "bad-handshake"):
            with self.subTest(mode=mode), self.assertRaises(TranscriptionError):
                ResidentBackend(str(self.worker), mode, startup_timeout=1)

    def test_empty_audio_does_not_send_a_shutdown_frame(self):
        backend = self.backend()
        self.assertEqual(backend.transcribe([], 0), "")
        self.assertEqual(backend.transcribe([np.zeros(0, dtype=np.int16)], 0), "")
        self.assertIsNone(backend._process.poll())

    def test_oversized_audio_is_rejected_before_writing(self):
        backend = self.backend()
        with self.assertRaisesRegex(TranscriptionError, "60 second"):
            backend.transcribe([np.zeros(16000*61, dtype=np.int16)], 61)
        self.assertIsNone(backend._process.poll())

    def test_closed_backend_cannot_spawn_another_process(self):
        backend = self.backend()
        backend.close()
        with self.assertRaisesRegex(TranscriptionError, "closed"):
            backend.transcribe([np.zeros(100, dtype=np.int16)], .01)
