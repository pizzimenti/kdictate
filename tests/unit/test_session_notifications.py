"""Recording warnings must disappear when their recording window ends."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import kdictate.core.daemon as daemon_module
from kdictate.constants import STATE_IDLE, STATE_RECORDING
from kdictate.core.daemon import DictationDaemon
from kdictate.runtime import RuntimePaths
from tests.unit.test_daemon import _DummyStream, _StubBackend, _make_config


class SessionNotificationTest(unittest.TestCase):
    """Exercise warning lifetime without sending desktop notifications."""

    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        paths = RuntimePaths(
            state_file=Path(tmpdir.name) / "state",
            last_text_file=Path(tmpdir.name) / "last",
        )
        self.daemon = DictationDaemon(
            _make_config(paths),
            backend=_StubBackend(),
            runtime_paths=paths,
            logger=logging.getLogger("kdictate.tests.session_notification"),
            stream_factory=lambda **kwargs: _DummyStream(),
            input_device_resolver=lambda: ("microphone", True),
            notify_error_fn=lambda summary, body: None,
        )
        self.daemon._vad_worker = lambda: None
        self.daemon._decode_worker = lambda: None
        # Cleanup runs in reverse order: shut down while subprocesses are
        # still stubbed, so a failed test cannot notify the real desktop.
        self.popen = self.enterContext(patch.object(daemon_module.subprocess, "Popen"))
        self.close = self.enterContext(patch.object(daemon_module.subprocess, "run"))
        self.enterContext(patch.object(
            daemon_module.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}",
        ))
        self.enterContext(patch.object(daemon_module, "set_default_source_volume"))
        self.enterContext(patch.object(daemon_module, "ensure_active_engine"))
        self.addCleanup(self.daemon.shutdown)

    def assert_warning_closed(self) -> None:
        self.close.assert_called_once()
        command = self.close.call_args.args[0]
        self.assertEqual(command[-2:], ["org.freedesktop.Notifications.CloseNotification", "42"])
        self.assertIn("org.freedesktop.Notifications", command)

    def drain_control_queue(self) -> None:
        done = threading.Event()
        self.daemon._control_queue.put(done.set)
        self.assertTrue(done.wait(timeout=1.0))

    def blocking_notification(self) -> threading.Event:
        """Simulate notify-send printing an ID and waiting for an action."""

        displayed = threading.Event()
        killed = threading.Event()
        proc = self.popen.return_value
        proc.poll.side_effect = lambda: 0 if killed.is_set() else None
        proc.kill.side_effect = killed.set

        def communicate(timeout):
            if killed.is_set():
                return b"42\n", None
            displayed.set()
            killed.wait(timeout=timeout)
            raise subprocess.TimeoutExpired("notify-send", timeout, output=b"42\n")

        proc.communicate.side_effect = communicate
        return displayed

    def start_prompt(self) -> threading.Thread:
        self.daemon._run_start_session()
        self.daemon._on_session_limit(self.daemon._session_watchdog_cancel)
        return self.daemon._session_prompt_thread

    def test_continue_closes_warning_and_rearms_watchdog(self) -> None:
        proc = self.popen.return_value
        proc.poll.return_value = 0
        proc.communicate.return_value = (b"42\ncontinue\n", None)
        self.daemon._run_start_session()
        original_cancel = self.daemon._session_watchdog_cancel
        self.daemon._on_session_limit(original_cancel)
        self.daemon._session_prompt_thread.join(timeout=1.0)

        self.assert_warning_closed()
        self.assertEqual(self.daemon.get_state(), STATE_RECORDING)
        self.assertIsNot(self.daemon._session_watchdog_cancel, original_cancel)
        self.assertTrue(original_cancel.is_set())
        command = self.popen.call_args.args[0]
        self.assertIn("--print-id", command)
        self.assertIn("--transient", command)
        self.assertIn("--action=continue=Continue", command)
        proc.kill.assert_not_called()

    def test_manual_stop_closes_warning_before_original_deadline(self) -> None:
        displayed = self.blocking_notification()
        thread = self.start_prompt()
        self.assertTrue(displayed.wait(timeout=1.0))

        self.daemon.request_stop()
        self.drain_control_queue()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(self.daemon.get_state(), STATE_IDLE)
        self.assert_warning_closed()
        self.popen.return_value.kill.assert_called_once()
        self.popen.return_value.stdout.close.assert_called_once()

    def test_deadline_closes_warning_and_auto_stops(self) -> None:
        self.blocking_notification()
        self.daemon.config = replace(self.daemon.config, session_confirm_timeout_s=0.02)
        thread = self.start_prompt()
        thread.join(timeout=1.0)
        self.drain_control_queue()

        self.assertFalse(thread.is_alive())
        self.assertEqual(self.daemon.get_state(), STATE_IDLE)
        self.assert_warning_closed()

    def test_dismissal_closes_warning_and_auto_stops(self) -> None:
        proc = self.popen.return_value
        proc.poll.return_value = 0
        proc.communicate.return_value = (b"42\n", None)
        thread = self.start_prompt()
        thread.join(timeout=1.0)
        self.drain_control_queue()

        self.assertEqual(self.daemon.get_state(), STATE_IDLE)
        self.assert_warning_closed()

    def test_shutdown_waits_for_warning_cleanup(self) -> None:
        displayed = self.blocking_notification()
        thread = self.start_prompt()
        self.assertTrue(displayed.wait(timeout=1.0))

        self.daemon.shutdown()

        self.assertFalse(thread.is_alive())
        self.assert_warning_closed()

    def test_cancelled_timer_does_not_display_warning(self) -> None:
        self.daemon._run_start_session()
        cancelled = self.daemon._session_watchdog_cancel
        self.daemon._run_stop_session()
        self.daemon._run_start_session()
        self.daemon._on_session_limit(cancelled)

        self.popen.assert_not_called()
        self.assertIsNone(self.daemon._session_prompt_thread)

    def test_stop_during_notification_creation_still_captures_id(self) -> None:
        cancelled = threading.Event()
        proc = self.popen.return_value
        proc.poll.return_value = None
        replies = iter((b"", b"42\n"))

        def communicate(timeout):
            cancelled.set()
            if proc.kill.called:
                return b"42\n", None
            raise subprocess.TimeoutExpired("notify-send", timeout, output=next(replies))

        proc.communicate.side_effect = communicate
        result = self.daemon._prompt_session_continuation(10.0, cancelled)

        self.assertFalse(result)
        self.assert_warning_closed()
        self.assertEqual(proc.communicate.call_count, 3)

    def test_old_answer_does_not_affect_restarted_recording(self) -> None:
        for continued in (False, True):
            with self.subTest(continued=continued):
                entered = threading.Event()
                answer = threading.Event()

                def prompt(timeout):
                    entered.set()
                    answer.wait(timeout=1.0)
                    return continued

                self.daemon._session_prompt_fn = prompt
                thread = self.start_prompt()
                self.assertTrue(entered.wait(timeout=1.0))
                self.daemon._run_stop_session()
                self.daemon._run_start_session()
                new_cancel = self.daemon._session_watchdog_cancel
                answer.set()
                thread.join(timeout=1.0)
                self.drain_control_queue()

                self.assertEqual(self.daemon.get_state(), STATE_RECORDING)
                self.assertIs(self.daemon._session_watchdog_cancel, new_cancel)
                self.assertFalse(new_cancel.is_set())
                self.daemon._run_stop_session()

    def test_queued_timeout_cannot_stop_a_new_session(self) -> None:
        entered = threading.Event()
        answer = threading.Event()
        control_blocked = threading.Event()
        release_control = threading.Event()
        self.addCleanup(release_control.set)

        def prompt(timeout):
            entered.set()
            answer.wait(timeout=1.0)
            return False

        def block_control():
            control_blocked.set()
            release_control.wait(timeout=2.0)

        self.daemon._session_prompt_fn = prompt
        thread = self.start_prompt()
        self.assertTrue(entered.wait(timeout=1.0))
        self.daemon._control_queue.put(block_control)
        self.assertTrue(control_blocked.wait(timeout=1.0))
        self.daemon.request_stop()
        self.daemon.request_start()
        answer.set()
        thread.join(timeout=1.0)
        release_control.set()
        self.drain_control_queue()

        self.assertEqual(self.daemon.get_state(), STATE_RECORDING)
        self.assertFalse(self.daemon._session_watchdog_cancel.is_set())

    def test_close_failure_does_not_prevent_auto_stop(self) -> None:
        proc = self.popen.return_value
        proc.poll.return_value = 0
        proc.communicate.return_value = (b"42\n", None)
        self.close.side_effect = subprocess.TimeoutExpired("gdbus", 2.0)
        thread = self.start_prompt()
        thread.join(timeout=1.0)
        self.drain_control_queue()

        self.assertEqual(self.daemon.get_state(), STATE_IDLE)
        self.assert_warning_closed()
