"""Long-lived Whisper dictation daemon.

The daemon keeps the speech model warm in memory and exposes a tiny signal-based
control plane:

- ``SIGUSR1`` starts recording.
- ``SIGUSR2`` stops recording and transcribes the captured audio.

Runtime state is persisted under ``XDG_RUNTIME_DIR`` so terminal helpers and the
Wayland hotkey listener can coordinate without importing audio or GI bindings.
Typing can happen here or be delegated to the hotkey listener with
``--no-type-output``.

During a PTT session, audio is transcribed in real-time: each utterance chunk
(committed by a silence gap or max-length limit) is decoded and typed immediately
rather than waiting for key release.
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from desktop_actions import notify, type_text
from dictate_runtime import (
    STATE_IDLE,
    STATE_RECORDING,
    STATE_TRANSCRIBING,
    RuntimePaths,
    default_runtime_paths,
    write_last_text,
    write_state,
)
from runtime_profile import recommended_shortform_cpu_threads, resolve_runtime, set_thread_env


DEFAULT_RUNTIME_PATHS = default_runtime_paths()
DEFAULT_MODEL_DIR = Path(__file__).parent / "models/distil-medium-en-ct2-int8"


def parse_args() -> argparse.Namespace:
    """Parse daemon configuration for model/runtime behavior."""

    parser = argparse.ArgumentParser(
        description="Whisper-Dictate daemon. SIGUSR1 starts recording, SIGUSR2 stops."
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the CTranslate2 model directory.",
    )
    parser.add_argument("--language", default="en", help="Language code for transcription.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate.")
    parser.add_argument("--beam-size", type=int, default=1, help="Whisper beam size.")
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition on previous text between segments.",
    )
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable built-in VAD filtering before decode.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Reject segments below this no-speech confidence threshold.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=recommended_shortform_cpu_threads(),
        help="Override CPU thread count. Defaults to a short-form latency-oriented value.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=("float32", "float16", "int8", "int8_float16"),
        help="Compute type. Defaults to int8 for the tuned CPU dictation path.",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_RUNTIME_PATHS.state_file),
        help="Path to the daemon state file used by control helpers.",
    )
    parser.add_argument(
        "--last-text-file",
        default=str(DEFAULT_RUNTIME_PATHS.last_text_file),
        help="Path to the latest transcript file used by control helpers.",
    )
    parser.add_argument(
        "--type-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Type the transcript into the focused window. Disable when an external helper owns typing.",
    )
    # VAD / streaming parameters (same defaults as mic_realtime.py)
    parser.add_argument(
        "--block-ms",
        type=int,
        default=30,
        help="Audio capture block duration in milliseconds.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=300.0,
        help="RMS threshold for speech detection. Increase to ignore noise.",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=220,
        help="Silence duration (ms) that commits the current utterance for transcription.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=180,
        help="Minimum speech duration (ms) required to transcribe an utterance.",
    )
    parser.add_argument(
        "--start-speech-ms",
        type=int,
        default=90,
        help="Consecutive voiced duration (ms) required before an utterance starts.",
    )
    parser.add_argument(
        "--max-utterance-s",
        type=float,
        default=2.5,
        help="Force-commit an utterance when it reaches this length in seconds.",
    )
    return parser.parse_args()


def _load_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Load the configured faster-whisper model and runtime profile."""

    runtime = resolve_runtime("cpu", args.compute_type, args.cpu_threads)
    set_thread_env(runtime["cpu_threads"])

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading model from {model_dir}...", flush=True)
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(model_dir),
        device=runtime["device"],
        compute_type=runtime["compute_type"],
        cpu_threads=runtime["cpu_threads"],
        num_workers=1,
    )
    return model, runtime


def _transcribe_pcm(model: Any, pcm_chunks: list[Any], args: argparse.Namespace) -> str:
    """Transcribe a list of int16 PCM chunks and return normalized text."""
    import numpy as np

    audio = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
    audio = audio.clip(-1.0, 1.0)
    segments, _ = model.transcribe(
        audio,
        language=args.language,
        beam_size=args.beam_size,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=args.no_speech_threshold,
        without_timestamps=True,
    )
    text = " ".join(s.text.strip() for s in segments if s.text and s.text.strip()).strip()
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


class DictationDaemon:
    """Own the warm model plus the record/transcribe lifecycle."""

    def __init__(self, args: argparse.Namespace, model: Any, runtime_paths: RuntimePaths) -> None:
        self.args = args
        self.model = model
        self.runtime_paths = runtime_paths
        self._lock = threading.Lock()
        self._recording = False
        self._transcribing = False
        self._stream: Any | None = None

        # Streaming pipeline state
        self._audio_queue: queue.Queue = queue.Queue(maxsize=512)
        self._utterance_queue: queue.Queue = queue.Queue(maxsize=64)
        self._stop_vad = threading.Event()
        self._vad_thread: threading.Thread | None = None
        self._decode_thread: threading.Thread | None = None
        self._streamed_text: list[str] = []

        self._set_runtime_state(STATE_IDLE)
        write_last_text(self.runtime_paths.last_text_file, "")

    def _set_runtime_state(self, value: str) -> None:
        write_state(self.runtime_paths.state_file, value)

    def _close_stream(self, stream: Any | None) -> None:
        """Stop and close a sounddevice stream if one is present."""

        if stream is None:
            return
        try:
            stream.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    def _input_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info, status
        with self._lock:
            if not self._recording:
                return
        chunk = indata[:, 0].copy()
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            pass  # drop block rather than stall the audio thread

    def _vad_worker(self) -> None:
        """Segment audio by silence/maxlen and post utterance chunks to the decode queue."""
        import numpy as np

        block_ms = self.args.block_ms
        sample_rate = self.args.sample_rate
        silence_blocks = max(1, int(self.args.silence_ms / block_ms))
        min_speech_blocks = max(1, int(self.args.min_speech_ms / block_ms))
        start_speech_blocks = max(1, int(self.args.start_speech_ms / block_ms))
        max_utterance_blocks = max(1, int((self.args.max_utterance_s * 1000.0) / block_ms))
        energy_threshold = self.args.energy_threshold

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
            if speech_block_count >= min_speech_blocks and utterance_pcm:
                audio_seconds = sum(len(c) for c in utterance_pcm) / float(sample_rate)
                try:
                    self._utterance_queue.put_nowait((list(utterance_pcm), audio_seconds))
                except queue.Full:
                    pass
            in_speech = False
            speech_block_count = 0
            pending_speech_block_count = 0
            trailing_silence_count = 0
            utterance_pcm.clear()
            pending_speech_pcm.clear()
            pending_silence_pcm.clear()

        while not self._stop_vad.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            voiced = rms >= energy_threshold

            if voiced:
                if not in_speech:
                    pending_speech_pcm.append(chunk)
                    pending_speech_block_count += 1
                    if pending_speech_block_count >= start_speech_blocks:
                        in_speech = True
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
                commit()
                continue

            if in_speech and trailing_silence_count >= silence_blocks:
                commit()

        # Flush any in-progress utterance when recording stops
        if in_speech and speech_block_count >= min_speech_blocks and utterance_pcm:
            commit()

        # Signal the decode thread to exit
        self._utterance_queue.put(None)

    def _decode_worker(self) -> None:
        """Transcribe each utterance chunk and type it immediately."""
        while True:
            item = self._utterance_queue.get()
            if item is None:
                break
            pcm_chunks, _audio_seconds = item
            try:
                text = _transcribe_pcm(self.model, pcm_chunks, self.args)
                if text:
                    with self._lock:
                        self._streamed_text.append(text)
                    if self.args.type_output:
                        type_text(text + " ")
                    print(f"Streamed: {text}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"Decode failed: {exc}", file=sys.stderr, flush=True)

    def start_recording(self) -> None:
        """Start the microphone stream and streaming VAD/decode pipeline."""

        import sounddevice as sd

        with self._lock:
            if self._recording:
                return
            if self._transcribing:
                notify("Still transcribing previous utterance.")
                return
            self._recording = True
            self._streamed_text = []
            # Drain any stale audio from a previous session
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            self._stop_vad.clear()
            self._set_runtime_state(STATE_RECORDING)
            write_last_text(self.runtime_paths.last_text_file, "")

        # Start decode thread first so it's ready when VAD posts utterances
        self._decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._decode_thread.start()

        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self._vad_thread.start()

        block_size = max(1, int(self.args.sample_rate * self.args.block_ms / 1000))
        stream: Any | None = None
        try:
            stream = sd.InputStream(
                samplerate=self.args.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                callback=self._input_callback,
            )
            with self._lock:
                self._stream = stream
            stream.start()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._recording = False
                self._stream = None
                self._set_runtime_state(STATE_IDLE)
            self._stop_vad.set()
            self._close_stream(stream)
            notify("Microphone start failed.")
            print(f"Recording start failed: {exc}", file=sys.stderr, flush=True)
            return

        print("Recording started (streaming).", flush=True)
        notify("● Listening...")

    def stop_and_transcribe(self) -> None:
        """Stop recording, flush the streaming pipeline, and finalize."""

        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._transcribing = True
            stream = self._stream
            self._stream = None
            self._set_runtime_state(STATE_TRANSCRIBING)

        self._close_stream(stream)

        # Signal VAD to flush remaining buffer and stop; it posts None to decode queue
        self._stop_vad.set()

        if self._vad_thread is not None:
            self._vad_thread.join(timeout=10)
            self._vad_thread = None

        if self._decode_thread is not None:
            self._decode_thread.join(timeout=10)
            self._decode_thread = None

        with self._lock:
            text = " ".join(self._streamed_text).strip()
            self._transcribing = False
            self._set_runtime_state(STATE_IDLE)

        if text:
            write_last_text(self.runtime_paths.last_text_file, text)
            preview = text[:60] + ("..." if len(text) > 60 else "")
            notify(f"✓ {preview}")
            print(f"Done: {text}", flush=True)
        else:
            write_last_text(self.runtime_paths.last_text_file, "")
            notify("No speech detected.")
            print("No speech detected.", flush=True)

    def request_start(self) -> None:
        """Queue a non-blocking start request from a signal handler."""

        threading.Thread(target=self.start_recording, daemon=True).start()

    def request_stop(self) -> None:
        """Queue a non-blocking stop/transcribe request from a signal handler."""

        threading.Thread(target=self.stop_and_transcribe, daemon=True).start()

    def install_signal_handlers(self) -> None:
        """Bind ``SIGUSR1``/``SIGUSR2`` to the daemon control actions."""

        def _on_sigusr1(signum: int, frame: Any) -> None:
            del signum, frame
            self.request_start()

        def _on_sigusr2(signum: int, frame: Any) -> None:
            del signum, frame
            self.request_stop()

        signal.signal(signal.SIGUSR1, _on_sigusr1)
        signal.signal(signal.SIGUSR2, _on_sigusr2)

    def shutdown(self) -> None:
        """Reset the daemon to idle and close any open microphone stream."""

        with self._lock:
            self._recording = False
            self._transcribing = False
            stream = self._stream
            self._stream = None

        self._close_stream(stream)
        self._stop_vad.set()
        self._set_runtime_state(STATE_IDLE)


def main() -> int:
    """Load the model, install signal handlers, and stay resident."""

    args = parse_args()

    try:
        model, runtime = _load_model(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("Run: python prepare_model.py", file=sys.stderr)
        return 1

    print(
        f"Model ready. device={runtime['device']} compute_type={runtime['compute_type']} "
        f"cpu_threads={runtime['cpu_threads']}",
        flush=True,
    )
    notify("Whisper-Dictate ready")

    daemon = DictationDaemon(
        args=args,
        model=model,
        runtime_paths=RuntimePaths(
            state_file=Path(args.state_file),
            last_text_file=Path(args.last_text_file),
        ),
    )
    daemon.install_signal_handlers()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
