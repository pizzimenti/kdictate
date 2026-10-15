"""Replay human speech through the daemon and real D-Bus/IBus controller.

Run under dbus-run-session. The capture device and IBus rendering surface are
replaced; VAD, native GPU inference, D-Bus transport and controller are real.
No microphone, desktop bus, gain change or active-engine change is used.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expect-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-text", required=True)
    parser.add_argument("--energy-threshold", type=float, default=0,
                        help="Zero keeps all blocks voiced to test preview without VAD pauses.")
    args = parser.parse_args()
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS", "").startswith("unix:path=/tmp/dbus-"):
        parser.error("run under dbus-run-session, away from the desktop bus")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_STATE_HOME"] = str(args.output_dir / "logs")
    os.environ["KDICTATE_STT_WORKER"] = str(args.worker.resolve())
    sys.path.insert(0, str(args.source_root.resolve()))

    import numpy as np
    from gi.repository import GLib
    from kdictate import backend
    from kdictate.config import DictationConfig, parse_args
    from kdictate.core import daemon as daemon_module
    from kdictate.ibus_engine.controller import DictationEngineController
    from kdictate.ibus_engine.dbus_client import DaemonControlBridge, DaemonSignalBridge
    from kdictate.service.dbus_service import SessionDbusService

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("kdictate.tests.native-preview")
    config = DictationConfig.from_namespace(parse_args([
        "--backend", "parakeet", "--state-file", str(args.output_dir / "state"),
        "--last-text-file", str(args.output_dir / "last-text"),
        "--session-max-recording-s", "0", "--energy-threshold", str(args.energy_threshold),
    ]))
    backend.PARAKEET_MODEL_PATH = args.model.resolve()
    model = backend.create_gpu_backend(config)
    assert model is not None, "native model initialization failed"
    worker = model._process

    # Suppress the two desktop mutations that normally precede capture.
    daemon_module.set_default_source_volume = lambda: True
    daemon_module.ensure_active_engine = lambda: None

    class ReplayStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self): pass
        def stop(self): pass
        def close(self): pass

    previews = []
    commits = []
    started = None

    class Adapter:
        def set_preedit(self, presentation):
            if presentation is not None and presentation.partial and started is not None:
                previews.append({"text": presentation.partial,
                                 "seconds": time.monotonic() - started,
                                 "state": controller.state.daemon_state})

        def commit_text(self, text):
            commits.append(text)

    context = GLib.MainContext.default()

    def pump_until(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while context.pending():
                context.iteration(False)
            if predicate():
                return
            time.sleep(.002)
        raise AssertionError("timed out waiting for recording/D-Bus events")

    daemon = daemon_module.DictationDaemon(
        config, model, config.runtime_paths, logger=logger,
        stream_factory=ReplayStream,
        input_device_resolver=lambda: ("recorded human speech", True),
        notify_error_fn=lambda *args: None,
    )
    service = SessionDbusService(daemon, logger=logger)
    controller = DictationEngineController(Adapter(), logger)
    signals = DaemonSignalBridge(controller, logger)
    control = DaemonControlBridge(logger)
    try:
        service.start()
        daemon.set_event_sink(service)
        controller.enable()
        controller.focus_in()
        signals.start()
        pump_until(lambda: controller.state.daemon_available)
        control.toggle()
        pump_until(lambda: controller.state.daemon_state == "recording")
        assert daemon.get_state() == "recording"
        with wave.open(str(args.audio), "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        frames = int(config.sample_rate * config.block_ms / 1000)
        started = time.monotonic()
        for offset in range(0, len(pcm), frames):
            block = pcm[offset:offset + frames]
            if len(block) < frames:
                block = np.pad(block, (0, frames - len(block)))
            daemon._handles.stream.callback(block[:, None], frames, None, None)
            next_block = started + (offset + frames) / config.sample_rate
            pump_until(lambda: time.monotonic() >= next_block)
        # Give the last captured block to VAD before issuing Stop.
        pump_until(lambda: daemon._audio_queue.empty())
        before_stop = [p for p in previews if p["state"] == "recording"]
        assert commits == [], "text was committed before Stop"
        stopped = time.monotonic()
        control.toggle()
        pump_until(lambda: controller.state.daemon_state == "idle" and bool(commits))
        assert commits == [args.expected_text], commits
        assert daemon.get_last_text() == args.expected_text
        assert model._process.pid == worker.pid, "model was reloaded during dictation"
        result = {
            "source_root": str(args.source_root.resolve()), "engine": "parakeet",
            "audio": str(args.audio.resolve()), "energy_threshold": args.energy_threshold,
            "audio_seconds": len(pcm) / config.sample_rate,
            "preview_before_stop": bool(before_stop),
            "first_preview_seconds": before_stop[0]["seconds"] if before_stop else None,
            "recording_previews": before_stop, "commits": commits,
            "stop_to_commit_seconds": time.monotonic() - stopped,
            "worker_reused": True,
        }
    finally:
        signals.stop()
        daemon.shutdown()
        service.stop()
    result["worker_reaped"] = worker.poll() is not None
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert result["worker_reaped"]
    assert result["preview_before_stop"] == args.expect_preview, result


if __name__ == "__main__":
    main()
