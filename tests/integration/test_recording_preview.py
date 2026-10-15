"""Exercise capture, VAD, decode, service events and the real IBus controller.

Only the audio device and model inference are replaced in these fast checks.
Unlike injected-transcript tests, no test calls a partial/final emit method.
"""
from __future__ import annotations

from dataclasses import replace
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from kdictate.config import DictationConfig, parse_args
from kdictate.core.daemon import DictationDaemon
from kdictate.ibus_engine.controller import DictationEngineController
from kdictate.runtime import read_last_text
from kdictate.service.dbus_service import SessionDbusService


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), "recording pipeline did not produce the expected event"


class Adapter:
    def __init__(self):
        self.preedits = []
        self.commits = []

    def set_preedit(self, presentation):
        self.preedits.append(presentation)

    def commit_text(self, text):
        self.commits.append(text)


class ScriptedBackend:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def transcribe(self, pcm_chunks, audio_seconds):
        self.calls.append(audio_seconds)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingHarness:
    def __init__(self, tmp_path, backend, *, engine="parakeet", **overrides):
        self.adapter = Adapter()
        logger = logging.getLogger("kdictate.tests.recording-preview")
        self.controller = DictationEngineController(self.adapter, logger)
        self.controller.enable()
        self.controller.focus_in()
        self.controller.set_daemon_available(True)
        config = DictationConfig.from_namespace(parse_args([
            "--backend", engine, "--state-file", str(tmp_path / "state"),
            "--last-text-file", str(tmp_path / "last"),
            "--session-max-recording-s", "0",
        ]))
        config = replace(config, **overrides)
        self.events = []

        def dispatch(name, args):
            self.events.append((name, args, self.daemon.get_state()))
            handlers = {
                "StateChanged": self.controller.handle_state_changed,
                "PartialTranscript": self.controller.handle_partial_transcript,
                "FinalTranscript": self.controller.handle_final_transcript,
                "ErrorOccurred": self.controller.handle_error,
            }
            handlers[name](*args)

        self.daemon = DictationDaemon(
            config, backend, config.runtime_paths, logger=logger,
            stream_factory=lambda **kw: SimpleNamespace(
                start=lambda: None, stop=lambda: None, close=lambda: None),
            input_device_resolver=lambda: ("recorded test audio", True),
            notify_error_fn=lambda *args: None,
        )
        self.daemon.set_event_sink(SessionDbusService(
            self.daemon, signal_sender=dispatch, logger=logger))
        self.daemon._run_start_session()
        assert self.daemon.get_state() == "recording"

    def feed(self, seconds, amplitude=2000):
        config = self.daemon.config
        frames = int(config.sample_rate * config.block_ms / 1000)
        block = np.full((frames, 1), amplitude, dtype=np.int16)
        for _ in range(round(seconds * 1000 / config.block_ms)):
            self.daemon._input_callback(block, frames, None, None)

    def has_preview(self, text):
        return any(p is not None and p.partial == text for p in self.adapter.preedits)

    def stop(self):
        self.daemon._run_stop_session()
        assert self.daemon.get_state() == "idle"


@pytest.fixture
def recording(tmp_path, monkeypatch):
    # Starting capture must not change the real mic gain or active IBus engine.
    monkeypatch.setattr("kdictate.core.daemon.set_default_source_volume", lambda: True)
    monkeypatch.setattr("kdictate.core.daemon.ensure_active_engine", lambda: None)
    sessions = []

    def create(backend, **kwargs):
        session = RecordingHarness(tmp_path, backend, **kwargs)
        sessions.append(session)
        return session

    yield create
    for session in sessions:
        session.daemon.shutdown()


def test_preview_before_stop_revises_draft_and_commits_only_final(recording):
    backend = ScriptedBackend("their", "there are", "there are two birds")
    session = recording(backend)
    session.feed(1.02)
    wait_for(lambda: session.has_preview("their"))
    assert session.daemon.get_state() == "recording"
    assert session.adapter.commits == []
    session.feed(1.02)
    wait_for(lambda: session.has_preview("there are"))
    assert not session.has_preview("their there are")
    assert session.adapter.commits == []
    session.stop()
    assert session.adapter.commits == ["there are two birds"]
    assert read_last_text(session.daemon.runtime_paths.last_text_file) == "there are two birds"
    assert len(backend.calls) == 3
    assert backend.calls[-1] == pytest.approx(2.04)


def test_next_utterance_preview_includes_previous_final_once(recording):
    backend = ScriptedBackend("first draft", "first sentence", "second draft", "second sentence")
    session = recording(backend)
    session.feed(1.02)
    wait_for(lambda: session.has_preview("first draft"))
    session.feed(.6, amplitude=0)
    wait_for(lambda: session.has_preview("first sentence"))
    session.feed(1.02)
    wait_for(lambda: session.has_preview("first sentence second draft"))
    assert session.adapter.commits == []
    session.stop()
    assert session.adapter.commits == ["first sentence second sentence"]


def test_empty_intermediate_decode_keeps_last_preview(recording):
    session = recording(ScriptedBackend("first draft", "", "revised draft", "whole sentence"))
    session.feed(1.02)
    wait_for(lambda: session.has_preview("first draft"))
    session.feed(1.02)
    wait_for(lambda: len(session.daemon._backend.calls) == 2)
    session.feed(1.02)
    # Observing the third response proves the empty second response has been
    # processed; checking the call count alone races the preview publication.
    wait_for(lambda: session.has_preview("revised draft"))
    session.stop()
    assert session.adapter.commits == ["whole sentence"]
    previews = [p.partial for p in session.adapter.preedits if p is not None]
    first = previews.index("first draft")
    assert "" not in previews[first:]


@pytest.mark.parametrize("last_response", ["", RuntimeError("final decode failed")])
def test_draft_cannot_become_final_on_empty_or_failed_final(recording, last_response):
    session = recording(ScriptedBackend("unconfirmed draft", last_response))
    session.feed(1.02)
    wait_for(lambda: session.has_preview("unconfirmed draft"))
    session.stop()
    assert session.adapter.commits == []
    assert session.daemon.get_last_text() == ""
    assert session.controller.state.pending_partial == ""


def test_failed_preview_allows_final_decode(recording):
    backend = ScriptedBackend(RuntimeError("preview failed"), "complete sentence")
    session = recording(backend)
    session.feed(1.02)
    wait_for(lambda: len(backend.calls) == 1)
    session.stop()
    assert session.adapter.commits == ["complete sentence"]
    assert not any(name == "ErrorOccurred" for name, _, _ in session.events)


def test_slow_preview_is_coalesced_and_stop_prioritizes_final(recording):
    started = threading.Event()
    release = threading.Event()

    class SlowBackend:
        calls = []

        def transcribe(self, pcm_chunks, audio_seconds):
            self.calls.append(audio_seconds)
            if len(self.calls) == 1:
                started.set()
                assert release.wait(timeout=3)
                return "old draft"
            return "whole sentence"

    backend = SlowBackend()
    session = recording(backend)
    try:
        session.feed(1.02)
        assert started.wait(timeout=3)
        session.feed(4.08)
        wait_for(lambda: session.daemon._audio_queue.empty())
        assert session.daemon._preview_queue.qsize() == 1
        stopper = threading.Thread(target=session.stop)
        stopper.start()
        wait_for(lambda: session.daemon.get_state() == "transcribing")
        release.set()
        stopper.join(timeout=3)
        assert not stopper.is_alive()
        assert session.adapter.commits == ["whole sentence"]
        assert len(backend.calls) == 2
        assert backend.calls[-1] == pytest.approx(5.1)
        assert not session.has_preview("old draft")
    finally:
        release.set()


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
def test_whisper_keeps_existing_utterance_decoding(recording, engine):
    backend = ScriptedBackend("unchanged final")
    session = recording(backend, engine=engine)
    session.feed(1.02)
    wait_for(lambda: session.daemon._audio_queue.empty())
    assert backend.calls == []
    session.stop()
    assert session.adapter.commits == ["unchanged final"]
