"""Tests for IBus active-engine restoration."""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from kdictate.core.ibus import KDICTATE_ENGINE_NAME, ensure_active_engine


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ibus"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class EnsureActiveEngineTest(unittest.TestCase):
    def test_returns_true_without_switching_when_engine_already_active(self) -> None:
        side_effect = [_completed(stdout=f"{KDICTATE_ENGINE_NAME}\n")]
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=side_effect) as run:
            self.assertTrue(ensure_active_engine())
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args_list[0].args[0], ["ibus", "engine"])

    def test_switches_engine_when_a_different_one_is_active(self) -> None:
        side_effect = [
            _completed(stdout="xkb:us::eng\n"),
            _completed(),
            _completed(stdout=f"{KDICTATE_ENGINE_NAME}\n"),
        ]
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=side_effect) as run:
            self.assertTrue(ensure_active_engine())
        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["ibus", "engine", KDICTATE_ENGINE_NAME],
        )

    def test_treats_nonzero_set_exit_as_success_when_verify_confirms(self) -> None:
        # `ibus engine <name>` is observed to occasionally exit non-zero
        # on KDE/Wayland even when the switch succeeds.
        side_effect = [
            _completed(stdout="xkb:us::eng\n"),
            _completed(returncode=1, stderr="warn"),
            _completed(stdout=f"{KDICTATE_ENGINE_NAME}\n"),
        ]
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=side_effect):
            self.assertTrue(ensure_active_engine())

    def test_returns_false_when_switch_does_not_stick(self) -> None:
        side_effect = [
            _completed(stdout="xkb:us::eng\n"),
            _completed(),
            _completed(stdout="xkb:us::eng\n"),
        ]
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=side_effect):
            self.assertFalse(ensure_active_engine())

    def test_returns_false_when_ibus_binary_is_missing(self) -> None:
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=OSError("no ibus")):
            self.assertFalse(ensure_active_engine())

    def test_returns_false_when_ibus_query_exits_nonzero(self) -> None:
        side_effect = [_completed(returncode=1, stderr="dbus error")]
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=side_effect):
            self.assertFalse(ensure_active_engine())

    def test_uses_utf8_decoding(self) -> None:
        side_effect = [_completed(stdout=f"{KDICTATE_ENGINE_NAME}\n")]
        with mock.patch("kdictate.core.ibus.subprocess.run", side_effect=side_effect) as run:
            ensure_active_engine()
        kwargs = run.call_args_list[0].kwargs
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["capture_output"], True)


if __name__ == "__main__":
    unittest.main()
