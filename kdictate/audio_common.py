"""Whisper model loading, transcription, and VAD helpers for the daemon."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("kdictate.daemon.vad")

VAD_QUEUE_POLL_TIMEOUT_S = 0.15
AUDIO_QUEUE_MAXSIZE = 512    # ~15s of 30ms blocks at 16kHz
UTTERANCE_QUEUE_MAXSIZE = 64  # max in-flight utterances

# Whisper models hallucinate these short phrases when the microphone
# captures ambient noise but no speech.  The filter is intentionally
# unconditional (no RMS or energy-level gate).
#
# Why not gate on audio energy?
#   The VAD only commits utterances whose per-block RMS already exceeds
#   energy_threshold (default 700).  Ambient mic noise in a typical room
#   produces avg_rms of 2000-4000 even during "silence", so every
#   committed utterance — including hallucinated ones — arrives with RMS
#   well above any useful suppression ceiling.  An RMS gate would simply
#   never fire.  (PR #9 tried and reverted this approach after daemon
#   logs confirmed the filter was unreachable.)
#
# Why is unconditional filtering safe?
#   The check only matches when the *entire* transcript is one of these
#   phrases.  A sentence containing "thank you" (e.g. "Thank you for
#   your help") passes through untouched.  The only false-positive risk
#   is a user dictating a standalone "okay" or "bye" — rare enough that
#   re-dictating is far less disruptive than phantom text appearing on
#   every silence gap.
HALLUCINATION_PHRASES: frozenset[str] = frozenset({
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "you",
    "bye",
    "goodbye",
    "the end",
    "thanks",
    "so",
    "okay",
})

#: Characters stripped before comparing against ``HALLUCINATION_PHRASES``.
_PUNCT_TABLE = str.maketrans("", "", ".,!?;:…\"'""''()[]{}")

def is_hallucination(text: str) -> bool:
    """Return True if *text* matches a known Whisper hallucination phrase."""
    normalized = " ".join(text.translate(_PUNCT_TABLE).strip().lower().split())
    return normalized in HALLUCINATION_PHRASES


def postprocess_transcript(text: str) -> str:
    """Normalize whitespace and suppress known Whisper hallucination phrases.

    Both backends should call this on raw transcript text before returning
    it to the daemon.  See the module-level comment above
    ``HALLUCINATION_PHRASES`` for why filtering is unconditional.
    """
    if not text:
        return ""
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if is_hallucination(text):
        logger.info("suppressed hallucination: %r", text)
        return ""
    return text


def load_whisper_model(
    model_dir: str | Path,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 1,
    num_workers: int = 1,
) -> Any:
    """Load a faster-whisper CTranslate2 model and return it.

    Import is deferred so callers that only need other helpers don't pay the
    import cost.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(model_dir),
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )


def transcribe_pcm(
    model: Any,
    pcm_chunks: list[Any],
    *,
    language: str = "en",
    task: str = "transcribe",
    beam_size: int = 1,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = False,
    vad_filter: bool = True,
) -> str:
    """Transcribe a list of int16 PCM chunks and return normalized text."""
    import numpy as np

    if not pcm_chunks:
        return ""

    audio = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
    audio = audio.clip(-1.0, 1.0)
    if audio.size == 0:
        return ""

    t0 = time.monotonic()
    segments, _ = model.transcribe(
        audio,
        language=language,
        task=task,
        beam_size=beam_size,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        no_speech_threshold=no_speech_threshold,
        without_timestamps=True,
    )
    seg_texts = [s.text.strip() for s in segments if s.text and s.text.strip()]
    elapsed = time.monotonic() - t0
    text = " ".join(seg_texts).strip()
    logger.info(
        "transcribe_pcm: %.1fs, %d segments, %d chars",
        elapsed, len(seg_texts), len(text),
    )
    if not text:
        return ""
    return postprocess_transcript(text)


# The noise floor is a low percentile of recent block energies rather than a
# running minimum or a decayed average.
#
# Why a percentile over a window, and why it updates unconditionally:
#   An estimator that only adapts *between* utterances cannot recover once
#   `in_speech` latches -- and `in_speech` is exactly what fails to clear when
#   the floor is wrong, so the error is self-sustaining. A percentile over a
#   trailing window has no such latch: pauses between words and syllables keep
#   landing in the low percentile even mid-utterance, so the estimate stays
#   anchored to the room rather than to the speaker.
#
# Window and percentile are chosen against each other: 10s of history at a
# 10th percentile means roughly one second of the quietest recent audio
# defines the floor. Natural speech always contains that much inter-word
# silence, so the floor does not chase the voice; sustained room noise has
# nothing quieter to offer, so the floor rises to meet it.
NOISE_WINDOW_BLOCKS: int = 333
NOISE_FLOOR_PERCENTILE: float = 10.0

# The adaptive gate stays disabled until the window holds this much audio, so
# the opening of a session is judged by the configured threshold alone rather
# than by an estimate derived from one or two blocks of the user's own voice.
NOISE_FLOOR_MIN_BLOCKS: int = 16

# Ceiling on how far the measured floor may push the gate above the configured
# threshold. Without it, a loud room can raise the gate past the user's own
# speech and reject an entire session -- and because the gate only ever moves
# up, `--energy-threshold` could not rescue it.
NOISE_FLOOR_MAX_MULTIPLE: float = 8.0


@dataclass
class VADConfig:
    """Parameters for the energy-based VAD segmenter.

    ``energy_threshold`` is the *lower bound* of the gate, not the whole gate.
    A fixed absolute RMS cannot separate speech from silence when the input
    gain is not fixed either -- and the daemon itself forces the mic to 91% on
    every activation (see ``core.audio.ACTIVATION_MIC_VOLUME_PERCENT``).
    Ambient room noise at that gain can measure an RMS of 2000-4000, several
    times the configured threshold, so every block scores as voiced:
    ``in_speech`` never ends, no silence gap is ever detected, and every
    utterance is a ``max_utterance_s`` force-commit chopped mid-word. Whisper
    then hallucinates over the noise-only fragments, which is what the
    HALLUCINATION_PHRASES filter above exists to mop up.

    The effective gate is therefore::

        clamp(noise_floor * noise_floor_margin,
              energy_threshold,
              energy_threshold * NOISE_FLOOR_MAX_MULTIPLE)

    with the floor measured from the audio itself. Both bounds matter:
    ``energy_threshold`` keeps a quiet room from driving the gate below what a
    weak microphone can produce, and the ceiling keeps a loud room from
    driving it above what the user's voice can produce.

    **The adaptive half is off by default** (``noise_floor_margin`` of 0), and
    the gate is then just ``energy_threshold``. Measured on real hardware, the
    trailing-window estimate tracked the *speaker* rather than the room:
    push-to-talk sessions are short and mostly speech, so the low percentile
    settled on quiet speech (~9000 RMS against an ~11000-13000 speaking level)
    with no silence in the window to anchor it. Both outcomes were bad — the
    gate either landed above the voice and rejected the session, or was
    clamped by the ceiling below the noise and passed every block, restoring
    the very never-ending-utterance bug it was added to fix.

    The floor is still measured and logged when the margin is 0, so the
    ``recording ended:`` line carries the data needed to choose one. Set a
    margin to opt in once it can be tuned against those logs.

    The default ``energy_threshold`` stays at the 700 that v0.13.0 chose for
    weak/quiet digital microphones.
    """

    sample_rate: int = 16000
    block_ms: int = 30
    energy_threshold: float = 700.0
    silence_ms: int = 600
    min_speech_ms: int = 120
    start_speech_ms: int = 90
    max_utterance_s: float = 10.0
    noise_floor_margin: float = 0.0

    @property
    def silence_blocks(self) -> int:
        return max(1, int(self.silence_ms / self.block_ms))

    @property
    def min_speech_blocks(self) -> int:
        return max(1, int(self.min_speech_ms / self.block_ms))

    @property
    def start_speech_blocks(self) -> int:
        return max(1, int(self.start_speech_ms / self.block_ms))

    @property
    def max_utterance_blocks(self) -> int:
        return max(1, int((self.max_utterance_s * 1000.0) / self.block_ms))


class VADSegmenter:
    """Energy-based voice activity detector that segments audio into utterances.

    Reads int16 PCM chunks from ``audio_queue`` and posts completed utterance
    chunk lists to ``utterance_queue``. Runs until ``stop_event`` is set, then
    flushes any in-progress utterance before posting a ``None`` sentinel.

    The utterance queue items are ``(pcm_chunks, audio_seconds)`` tuples, or
    ``None`` as the stop sentinel.
    """

    def __init__(
        self,
        config: VADConfig,
        audio_queue: queue.Queue,
        utterance_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        self.config = config
        self.audio_queue = audio_queue
        self.utterance_queue = utterance_queue
        self.stop_event = stop_event

    def run(self) -> None:
        """Block until stop_event is set, segmenting audio the whole time."""
        import numpy as np

        cfg = self.config
        silence_blocks = cfg.silence_blocks
        min_speech_blocks = cfg.min_speech_blocks
        start_speech_blocks = cfg.start_speech_blocks
        max_utterance_blocks = cfg.max_utterance_blocks

        logger.info(
            "vad config: energy_threshold=%.0f, noise_floor_margin=%.2f, "
            "start_speech_blocks=%d, min_speech_blocks=%d, silence_blocks=%d, "
            "sample_rate=%d, block_ms=%.0f",
            cfg.energy_threshold, cfg.noise_floor_margin, start_speech_blocks,
            min_speech_blocks, silence_blocks, cfg.sample_rate, cfg.block_ms,
        )
        session_start = time.monotonic()
        total_blocks = 0
        voiced_blocks = 0
        commits = 0
        peak_rms_overall = 0.0
        peak_rms_below_thresh = 0.0
        # The quietest block of the session. peak_below_gate goes to 0 the
        # moment nothing falls under the gate, which is exactly the case
        # where the gate is wrong and the number is needed most; this one is
        # measured independently of it.
        min_rms = float("inf")

        # Trailing history of block energies. The gate for each block is
        # derived from the blocks *before* it -- never from the block being
        # judged, which would be circular: with the floor seeded from the
        # current block, `rms >= rms * margin` is false for every rms, so the
        # first block of a session could never be voiced and the opening
        # syllable of push-to-talk dictation was dropped.
        noise_history: deque[float] = deque(maxlen=NOISE_WINDOW_BLOCKS)
        noise_floor = 0.0
        threshold = cfg.energy_threshold
        threshold_min = threshold
        threshold_max = threshold
        gate_ceiling = cfg.energy_threshold * NOISE_FLOOR_MAX_MULTIPLE

        # Opening blocks are withheld until the floor is measurable, then
        # replayed through the calibrated gate rather than judged as they
        # arrive. Judging them live would have to guess: score them against
        # the bare `energy_threshold` and a noisy room latches a spurious
        # noise-only utterance before the floor is ever known (precisely the
        # input Whisper hallucinates over), but skip them and a user who
        # speaks the instant the hotkey is pressed loses the opening
        # syllable. Replaying costs NOISE_FLOOR_MIN_BLOCKS of latency once
        # per session and needs neither guess.
        primed = cfg.noise_floor_margin <= 0
        prime_buffer: list[tuple[Any, float]] = []

        utterance_pcm: list[Any] = []
        pending_speech_pcm: list[Any] = []
        pending_silence_pcm: list[Any] = []
        in_speech = False
        speech_block_count = 0
        pending_speech_block_count = 0
        trailing_silence_count = 0

        def commit() -> None:
            nonlocal in_speech, speech_block_count, pending_speech_block_count
            nonlocal trailing_silence_count, utterance_pcm, pending_speech_pcm, pending_silence_pcm
            nonlocal commits
            if speech_block_count >= min_speech_blocks and utterance_pcm:
                audio_seconds = sum(len(c) for c in utterance_pcm) / float(cfg.sample_rate)
                avg_rms = float(np.sqrt(np.mean(
                    np.concatenate(utterance_pcm).astype(np.float32) ** 2
                )))
                pending = self.utterance_queue.qsize()
                try:
                    self.utterance_queue.put_nowait((list(utterance_pcm), audio_seconds))
                    commits += 1
                    logger.info(
                        "utterance committed: %.1fs audio, %d blocks, %d queued, avg_rms=%.0f",
                        audio_seconds, speech_block_count, pending, avg_rms,
                    )
                except queue.Full:
                    logger.warning("utterance dropped (queue full): %.1fs audio", audio_seconds)
            in_speech = False
            speech_block_count = 0
            pending_speech_block_count = 0
            trailing_silence_count = 0
            utterance_pcm.clear()
            pending_speech_pcm.clear()
            pending_silence_pcm.clear()

        def current_gate() -> float:
            """Recompute the gate from the noise history collected so far.

            The floor is measured even when ``noise_floor_margin`` is 0 and it
            cannot move the gate. That measurement is the only record of what
            this microphone in this room actually does, and it is what the
            margin has to be tuned against — a diagnostic that switches itself
            off along with the feature is no use for turning the feature back
            on.
            """

            nonlocal noise_floor, threshold, threshold_min, threshold_max
            if not noise_history:
                return threshold
            noise_floor = float(
                np.percentile(noise_history, NOISE_FLOOR_PERCENTILE)
            )
            if cfg.noise_floor_margin > 0:
                threshold = min(
                    max(
                        cfg.energy_threshold,
                        noise_floor * cfg.noise_floor_margin,
                    ),
                    gate_ceiling,
                )
                threshold_min = min(threshold_min, threshold)
                threshold_max = max(threshold_max, threshold)
            return threshold

        def handle_block(chunk: Any, rms: float, voiced: bool) -> None:
            """Advance the speech state machine by one already-scored block."""

            nonlocal total_blocks, voiced_blocks, peak_rms_overall
            nonlocal peak_rms_below_thresh, in_speech, speech_block_count
            nonlocal pending_speech_block_count, trailing_silence_count
            nonlocal utterance_pcm, pending_speech_pcm, pending_silence_pcm
            nonlocal min_rms

            total_blocks += 1
            if rms > peak_rms_overall:
                peak_rms_overall = rms
            if rms < min_rms:
                min_rms = rms
            if voiced:
                voiced_blocks += 1
            elif rms > peak_rms_below_thresh:
                peak_rms_below_thresh = rms

            if voiced:
                if not in_speech:
                    pending_speech_pcm.append(chunk)
                    pending_speech_block_count += 1
                    if pending_speech_block_count >= start_speech_blocks:
                        in_speech = True
                        logger.info("speech started (rms=%.0f)", rms)
                        utterance_pcm = list(pending_speech_pcm)
                        speech_block_count = len(utterance_pcm)
                        pending_speech_pcm = []
                        pending_speech_block_count = 0
                        pending_silence_pcm = []
                        trailing_silence_count = 0
                else:
                    if pending_silence_pcm:
                        utterance_pcm.extend(pending_silence_pcm)
                        pending_silence_pcm = []
                    utterance_pcm.append(chunk)
                    speech_block_count += 1
                    trailing_silence_count = 0
            elif in_speech:
                pending_silence_pcm.append(chunk)
                trailing_silence_count += 1
            else:
                pending_speech_pcm = []
                pending_speech_block_count = 0

            if in_speech and speech_block_count >= max_utterance_blocks:
                logger.info("force-commit (max utterance reached)")
                commit()
                return

            if in_speech and trailing_silence_count >= silence_blocks:
                commit()

        try:
            while not self.stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=VAD_QUEUE_POLL_TIMEOUT_S)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

                # A degenerate block (PortAudio can deliver a zero-length one
                # on an xrun) yields NaN, which compares False against every
                # bound. Left in the history it would poison the percentile
                # for the rest of the session, so it is scored as silence and
                # never recorded.
                finite = math.isfinite(rms)
                if not finite:
                    rms = 0.0
                else:
                    # Appended unconditionally, including mid-utterance. An
                    # estimator that only adapts while `not in_speech` cannot
                    # recover from a floor that is too low, because a too-low
                    # floor is precisely what keeps `in_speech` latched --
                    # the error would sustain itself for the whole session.
                    noise_history.append(rms)

                if not primed:
                    prime_buffer.append((chunk, rms))
                    if len(noise_history) < NOISE_FLOOR_MIN_BLOCKS:
                        continue
                    primed = True
                    gate = current_gate()
                    for held_chunk, held_rms in prime_buffer:
                        handle_block(held_chunk, held_rms, held_rms >= gate)
                    prime_buffer.clear()
                    continue

                handle_block(chunk, rms, rms >= current_gate())

            # Flush any in-progress utterance when recording stops
            if in_speech and speech_block_count >= min_speech_blocks and utterance_pcm:
                commit()
        finally:
            # Single-line per-session summary so a silent recording is
            # diagnosable from the log alone: peak_rms vs. the gate tells you
            # whether the mic gain is the issue, voiced_blocks vs. commits
            # tells you whether VAD heuristics rejected real speech.
            #
            # The gate is reported as the range it actually spanned, not as
            # its final value. `voiced_blocks` accumulates against whatever
            # gate was in force at each block, so printing only the last one
            # can produce a self-contradictory line -- blocks counted voiced
            # against a number the peak RMS never reached -- and a user
            # tuning `noise_floor_margin` against that reads it backwards.
            elapsed = time.monotonic() - session_start
            logger.info(
                "recording ended: %.1fs, %d blocks, %d voiced, %d committed, "
                "rms=%.0f-%.0f, peak_below_gate=%.0f, gate=%.0f-%.0f, "
                "noise_floor=%.0f, margin=%.2f",
                elapsed, total_blocks, voiced_blocks, commits,
                0.0 if min_rms == float("inf") else min_rms, peak_rms_overall,
                peak_rms_below_thresh, threshold_min, threshold_max,
                noise_floor, cfg.noise_floor_margin,
            )

            # A session that heard nothing is the one failure the summary
            # above is too terse to explain, and the fix depends on which
            # bound rejected the audio -- so name the knob rather than
            # leaving the user to infer it from raw numbers.
            if total_blocks and not voiced_blocks:
                logger.warning(
                    "no speech detected: %d blocks all below the %.0f-%.0f gate "
                    "(peak_rms=%.0f). If you were speaking, lower "
                    "--noise-floor-margin (currently %.2f) or "
                    "--energy-threshold (currently %.0f).",
                    total_blocks, threshold_min, threshold_max,
                    peak_rms_overall, cfg.noise_floor_margin,
                    cfg.energy_threshold,
                )

            # Always post the stop sentinel — even if the loop above raised
            # — so the decode consumer can never wedge waiting for a sentinel
            # that never arrives. A short timeout guards against the unlikely
            # case of a fully-saturated utterance_queue.
            try:
                self.utterance_queue.put(None, timeout=1.0)
                logger.info("vad sentinel posted")
            except Exception:  # noqa: BLE001
                logger.warning("vad sentinel post failed (queue full or timeout)")
