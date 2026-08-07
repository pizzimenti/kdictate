"""Regression tests for the adaptive-noise-floor VAD segmenter.

Each test here pins a failure mode the segmenter has actually shipped with.
Blocks are constant-amplitude int16, so a block filled with ``v`` has an RMS
of exactly ``v`` and the arithmetic under test is fully deterministic.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

import numpy as np

from kdictate.audio_common import (
    NOISE_FLOOR_MAX_MULTIPLE,
    VADConfig,
    VADSegmenter,
)
from kdictate.daemon_profiles import daemon_arg_defaults

BLOCK_SAMPLES = 480  # 30ms at 16kHz


def _block(rms: int) -> np.ndarray:
    """A block whose RMS is exactly *rms*."""

    return np.full(BLOCK_SAMPLES, rms, dtype=np.int16)


def _degenerate_block() -> np.ndarray:
    """A zero-length block, as PortAudio can deliver on an xrun (RMS -> NaN)."""

    return np.array([], dtype=np.int16)


def _run_vad(blocks: list[np.ndarray], **overrides: object) -> list[tuple]:
    """Feed *blocks* through a segmenter and return the committed utterances."""

    cfg = VADConfig(**overrides)  # type: ignore[arg-type]
    audio_q: queue.Queue = queue.Queue()
    utt_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    segmenter = VADSegmenter(cfg, audio_q, utt_q, stop)

    worker = threading.Thread(target=segmenter.run, daemon=True)
    worker.start()
    for block in blocks:
        audio_q.put(block)

    deadline = time.monotonic() + 10.0
    while not audio_q.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.1)  # let the final block finish processing
    stop.set()
    worker.join(timeout=10.0)

    committed = []
    while True:
        try:
            item = utt_q.get_nowait()
        except queue.Empty:
            break
        if item is not None:
            committed.append(item)
    return committed


class FirstUtteranceTest(unittest.TestCase):
    def test_speech_from_the_very_first_block_is_captured(self) -> None:
        """The gate must never be derived from the block it is judging.

        Seeding the noise floor from block 0 made `rms >= rms * margin` the
        test for that block, which is false for every rms > 0. Because the
        floor then had nothing quieter to fall to, the whole run stayed
        un-voiced and nothing was committed at all.
        """

        blocks = [_block(6000)] * 40 + [_block(100)] * 30
        committed = _run_vad(blocks)

        self.assertEqual(len(committed), 1)
        pcm, seconds = committed[0]
        # All 40 speech blocks, including the ones before the latch.
        self.assertGreaterEqual(sum(len(c) for c in pcm), 40 * BLOCK_SAMPLES)
        self.assertGreater(seconds, 0.0)


class NoisyRoomTest(unittest.TestCase):
    """Ambient well above `energy_threshold` must still yield silence gaps."""

    def test_utterance_ends_on_silence_when_ambient_exceeds_energy_threshold(self) -> None:
        # Ambient at 3000 is >4x the 700 floor: under a fixed threshold every
        # ambient block scores as voiced, no silence gap is ever found, and
        # the utterance runs to max_utterance_s and is chopped mid-word.
        blocks = [_block(3000)] * 60 + [_block(12000)] * 100 + [_block(3000)] * 60
        committed = _run_vad(blocks, noise_floor_margin=1.6)

        self.assertEqual(len(committed), 1)
        # Committed by the silence gap, well short of max_utterance_s.
        _pcm, seconds = committed[0]
        self.assertLess(seconds, 5.0)

    def test_fixed_threshold_mode_still_available(self) -> None:
        # margin=0 disables the adaptive middle. Same audio, and now the
        # ambient never falls below the gate, so no silence gap is found and
        # the run only ends via the stop-flush.
        blocks = [_block(3000)] * 60 + [_block(12000)] * 100 + [_block(3000)] * 60
        committed = _run_vad(blocks, noise_floor_margin=0.0)

        self.assertEqual(len(committed), 1)
        _pcm, seconds = committed[0]
        self.assertGreater(seconds, 5.0)


class GateCeilingTest(unittest.TestCase):
    def test_loud_room_cannot_push_the_gate_past_the_users_voice(self) -> None:
        """The adaptive gate is clamped, so a loud room cannot mute a session.

        Ambient 5000 * margin 1.6 = 8000 would sit above the 6000 speech and
        reject the entire recording. The ceiling (energy_threshold * 8 =
        5600) keeps the gate below the voice.
        """

        energy_threshold = 700.0
        self.assertGreater(5000 * 1.6, energy_threshold * NOISE_FLOOR_MAX_MULTIPLE)

        blocks = [_block(5000)] * 60 + [_block(6000)] * 100 + [_block(5000)] * 60
        committed = _run_vad(
            blocks, energy_threshold=energy_threshold, noise_floor_margin=1.6,
        )

        self.assertEqual(len(committed), 1)


class AdaptiveGateDefaultTest(unittest.TestCase):
    def test_adaptive_gate_is_off_unless_asked_for(self) -> None:
        """The gate defaults to the fixed threshold.

        Measured on real hardware the trailing-window estimate tracked the
        speaker rather than the room, so the adaptive half is opt-in until it
        can be tuned against real logs. Pinned here because the default is the
        whole of the decision — the mechanism itself still works when asked.
        """

        self.assertEqual(VADConfig().noise_floor_margin, 0.0)
        self.assertEqual(daemon_arg_defaults()["noise_floor_margin"], 0.0)


class DegenerateBlockTest(unittest.TestCase):
    def test_nan_block_does_not_poison_the_noise_floor(self) -> None:
        """One NaN must not silently disable the adaptive gate for the session.

        NaN compares False against every bound, so an unfiltered NaN stuck in
        the estimator collapsed the gate back to `energy_threshold` for the
        rest of the recording -- restoring exactly the never-ending-utterance
        behavior the adaptive gate exists to prevent.
        """

        blocks = (
            [_block(3000)] * 40
            + [_degenerate_block()]
            + [_block(12000)] * 100
            + [_block(3000)] * 40
        )
        with np.errstate(invalid="ignore"):
            committed = _run_vad(blocks, noise_floor_margin=1.6)

        self.assertEqual(len(committed), 1)
        _pcm, seconds = committed[0]
        self.assertLess(seconds, 5.0)


if __name__ == "__main__":
    unittest.main()
