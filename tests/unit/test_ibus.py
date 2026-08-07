"""Tests for IBus active-engine restoration."""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from kdictate.core import ibus as ibus_module
from kdictate.core.ibus import KDICTATE_ENGINE_NAME, ensure_active_engine


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ibus"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class EnsureActiveEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        # Two module-level hazards these tests would otherwise inherit:
        #
        #   - The display-env cache is process-global. Leaving it populated
        #     (or poisoned with {}) leaks into later tests, so *which* tests
        #     fail depends on collection order rather than on the code.
        #   - `_display_env` shells out to `systemctl` through the very
        #     `subprocess.run` these tests patch with fixed side_effect
        #     lists, so on a host with no WAYLAND_DISPLAY it silently eats
        #     the entry meant for `ibus engine`.
        #
        # Pinning the resolved env keeps the assertions below about the
        # `ibus` calls alone; DisplayEnvTest covers the resolution itself.
        ibus_module._reset_display_env_cache()
        self.addCleanup(ibus_module._reset_display_env_cache)
        patcher = mock.patch.object(ibus_module, "_display_env", return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

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


class DisplayEnvTest(unittest.TestCase):
    """Resolution of the display variables the ``ibus`` CLI needs.

    ``ibus`` picks its bus socket by display name rather than over D-Bus, so
    a missing or stale WAYLAND_DISPLAY makes every call fail against a socket
    file that belongs to no live session.
    """

    SHOW_ENVIRONMENT = "LANG=en_US.UTF-8\nWAYLAND_DISPLAY=wayland-1\nDISPLAY=:1\n"

    def setUp(self) -> None:
        ibus_module._reset_display_env_cache()
        self.addCleanup(ibus_module._reset_display_env_cache)

    def test_uses_process_environment_without_querying_systemd(self) -> None:
        with mock.patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
            with mock.patch("kdictate.core.ibus.subprocess.run") as run:
                self.assertEqual(
                    ibus_module._display_env()["WAYLAND_DISPLAY"], "wayland-0"
                )
        run.assert_not_called()

    def test_falls_back_to_systemd_when_wayland_display_missing(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "WAYLAND_DISPLAY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "kdictate.core.ibus.subprocess.run",
                return_value=_completed(stdout=self.SHOW_ENVIRONMENT),
            ) as run:
                resolved = ibus_module._display_env()
        self.assertEqual(resolved["WAYLAND_DISPLAY"], "wayland-1")
        self.assertEqual(run.call_args.args[0], ["systemctl", "--user", "show-environment"])

    def test_heal_prefers_systemd_over_stale_process_environment(self) -> None:
        # A daemon that outlived its session still has a WAYLAND_DISPLAY in
        # os.environ, pointing at a compositor that is gone. If the normal
        # environment-first precedence survived the reset, the retry would
        # keep re-resolving the same dead display and never converge.
        with mock.patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
            with mock.patch("kdictate.core.ibus.subprocess.run") as run:
                self.assertEqual(
                    ibus_module._display_env()["WAYLAND_DISPLAY"], "wayland-0"
                )
                run.assert_not_called()

            # Mirrors what ensure_active_engine does when a query comes back
            # None; the plain reset used elsewhere must NOT flip precedence.
            ibus_module._reset_display_env_cache(prefer_systemd=True)

            with mock.patch(
                "kdictate.core.ibus.subprocess.run",
                return_value=_completed(stdout=self.SHOW_ENVIRONMENT),
            ):
                healed = ibus_module._display_env()
        self.assertEqual(healed["WAYLAND_DISPLAY"], "wayland-1")

    def test_run_ibus_passes_resolved_display_env(self) -> None:
        with mock.patch.object(
            ibus_module, "_display_env", return_value={"WAYLAND_DISPLAY": "wayland-7"}
        ):
            with mock.patch(
                "kdictate.core.ibus.subprocess.run", return_value=_completed()
            ) as run:
                ibus_module._run_ibus("engine")
        passed = run.call_args.kwargs["env"]
        self.assertEqual(passed["WAYLAND_DISPLAY"], "wayland-7")
        # A full environment, not just the overrides -- ibus still needs
        # DBUS_SESSION_BUS_ADDRESS and friends to reach the bus.
        self.assertIn("PATH", passed)


if __name__ == "__main__":
    unittest.main()
