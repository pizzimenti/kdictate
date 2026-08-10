"""Tests for the Python installer helpers."""

from __future__ import annotations

import io
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import install
from kdictate.constants import DBUS_INTERFACE


class InstallHelperTests(unittest.TestCase):
    def test_next_preload_engines_returns_none_when_engine_is_present(self) -> None:
        current = f"['xkb:de::ger', '{DBUS_INTERFACE}']"
        self.assertIsNone(install.next_preload_engines(current, DBUS_INTERFACE))

    def test_next_preload_engines_adds_only_kdictate_for_typed_empty(self) -> None:
        self.assertEqual(
            install.next_preload_engines("@as []", DBUS_INTERFACE),
            f"['{DBUS_INTERFACE}']",
        )

    def test_next_preload_engines_adds_only_kdictate_for_bare_empty(self) -> None:
        self.assertEqual(
            install.next_preload_engines("[]", DBUS_INTERFACE),
            f"['{DBUS_INTERFACE}']",
        )

    def test_next_preload_engines_adds_only_kdictate_for_empty_string(self) -> None:
        self.assertEqual(
            install.next_preload_engines("", DBUS_INTERFACE),
            f"['{DBUS_INTERFACE}']",
        )

    def test_next_preload_engines_strips_typed_prefix_before_append(self) -> None:
        self.assertEqual(
            install.next_preload_engines("@as ['xkb:de::ger']", DBUS_INTERFACE),
            f"['xkb:de::ger', '{DBUS_INTERFACE}']",
        )

    def test_next_preload_engines_rejects_unexpected_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected preload-engines value"):
            install.next_preload_engines("not-a-list", DBUS_INTERFACE)

    # -- previous_preload_engines (the inverse) ------------------------------

    def test_previous_preload_engines_returns_none_when_engine_absent(self) -> None:
        self.assertIsNone(
            install.previous_preload_engines("['xkb:de::ger']", DBUS_INTERFACE)
        )

    def test_previous_preload_engines_returns_none_when_list_empty(self) -> None:
        self.assertIsNone(install.previous_preload_engines("@as []", DBUS_INTERFACE))
        self.assertIsNone(install.previous_preload_engines("[]", DBUS_INTERFACE))
        self.assertIsNone(install.previous_preload_engines("", DBUS_INTERFACE))

    def test_previous_preload_engines_removes_kdictate_only_entry(self) -> None:
        current = f"['{DBUS_INTERFACE}']"
        self.assertEqual(
            install.previous_preload_engines(current, DBUS_INTERFACE),
            "@as []",
        )

    def test_previous_preload_engines_removes_kdictate_first_of_two(self) -> None:
        current = f"['{DBUS_INTERFACE}', 'xkb:de::ger']"
        self.assertEqual(
            install.previous_preload_engines(current, DBUS_INTERFACE),
            "['xkb:de::ger']",
        )

    def test_previous_preload_engines_removes_kdictate_last_of_two(self) -> None:
        current = f"['xkb:de::ger', '{DBUS_INTERFACE}']"
        self.assertEqual(
            install.previous_preload_engines(current, DBUS_INTERFACE),
            "['xkb:de::ger']",
        )

    def test_previous_preload_engines_removes_kdictate_middle_of_three(self) -> None:
        current = f"['xkb:de::ger', '{DBUS_INTERFACE}', 'ibus-anthy']"
        self.assertEqual(
            install.previous_preload_engines(current, DBUS_INTERFACE),
            "['xkb:de::ger', 'ibus-anthy']",
        )

    def test_previous_preload_engines_strips_typed_prefix(self) -> None:
        current = f"@as ['xkb:de::ger', '{DBUS_INTERFACE}']"
        self.assertEqual(
            install.previous_preload_engines(current, DBUS_INTERFACE),
            "['xkb:de::ger']",
        )

    def test_previous_preload_engines_rejects_unexpected_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected preload-engines value"):
            install.previous_preload_engines(
                f"not-a-list-but-contains-'{DBUS_INTERFACE}'",
                DBUS_INTERFACE,
            )

    def test_next_then_previous_round_trips_existing_list(self) -> None:
        original = "['xkb:de::ger', 'ibus-anthy']"
        added = install.next_preload_engines(original, DBUS_INTERFACE)
        self.assertIsNotNone(added)
        removed = install.previous_preload_engines(added, DBUS_INTERFACE)
        self.assertEqual(removed, original)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["pacman"], returncode=returncode, stdout=stdout, stderr="",
    )


def _ctx(runtime_dir: Path) -> install.InstallContext:
    return install.InstallContext(
        script_path=Path("/tmp/install.py"),
        script_dir=Path("/tmp"),
        home=Path("/tmp/home"),
        runtime_dir=runtime_dir,
    )


class InstalledVersionTests(unittest.TestCase):
    def test_packaged_version_comes_from_pacman_without_the_pkgrel(self) -> None:
        # The pkgrel is not part of the project's version lockstep, so it has
        # to be stripped or the comparison against __version__ never matches
        # and an up-to-date system would reinstall on every run.
        with mock.patch.object(
            install, "run_command", return_value=_completed("kdictate 0.13.0-1\n")
        ):
            self.assertEqual(
                install._installed_version(_ctx(Path("/nonexistent")), True), "0.13.0"
            )

    def test_packaged_version_is_none_when_the_package_is_absent(self) -> None:
        with mock.patch.object(
            install, "run_command", return_value=_completed("", returncode=1)
        ):
            self.assertIsNone(
                install._installed_version(_ctx(Path("/nonexistent")), True)
            )

    def test_source_version_is_none_without_a_runtime_venv(self) -> None:
        self.assertIsNone(
            install._installed_version(_ctx(Path("/nonexistent")), False)
        )

    def test_source_probe_cannot_import_the_tree_it_is_installing_from(self) -> None:
        """The probe must ask about the *installed* copy, not this checkout.

        `python -c` puts the cwd on sys.path, and the documented invocation is
        `python3 install.py` from the repo root — which contains the kdictate
        package. Without -P the probe imported the source tree, always
        reported its own version, and made the version gate a no-op that could
        never detect an update.
        """

        ctx = _ctx(Path("/tmp/kdictate-runtime"))
        completed = _completed("0.12.0\n")
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(
                install, "run_command", return_value=completed
            ) as run:
                self.assertEqual(install._installed_version(ctx, False), "0.12.0")

        argv = run.call_args.args[0]
        # -I, not -P: -P only stops the cwd being prepended, while PYTHONPATH
        # is still honoured, and a dev shell or IDE may well have this very
        # checkout on it. -I implies -P, -E and -s.
        self.assertIn("-I", argv)
        # ...and not resolved relative to wherever the installer was invoked.
        self.assertEqual(run.call_args.kwargs["cwd"], ctx.home)


class BuildDependencyProbeTests(unittest.TestCase):
    def test_probe_cannot_be_fooled_by_a_stray_build_directory(self) -> None:
        """The probe must not import directories from the invocation cwd.

        This runs from a repo root that accumulates a `build/` directory from
        the very wheel step these dependencies exist to perform. `python -c`
        puts the cwd on sys.path, so Python imports that directory as an
        implicit namespace package and `import build` succeeds even when
        python-build is not installed at all — reporting only *some* of the
        missing packages, so the user installs those, re-runs, and makepkg
        dies at `python -m build` anyway.
        """

        ok = subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr="")
        with mock.patch.object(install, "run_command", return_value=ok) as run:
            install._require_build_dependencies()

        self.assertTrue(run.call_args_list, "probe never ran")
        for call in run.call_args_list:
            # -I rather than -P: -P leaves PYTHONPATH honoured, so a shell
            # exporting this checkout would still shadow the real module.
            self.assertIn("-I", call.args[0])
            # The specific directory matters. Accepting any non-null cwd would
            # pass even if the probe ran inside the checkout it must avoid.
            self.assertEqual(call.kwargs.get("cwd"), Path.home())


class GlobalShortcutRepairTests(unittest.TestCase):
    """--reconfigure must be able to repair a broken Ctrl+Space binding."""

    ENTRY = "_launch=Ctrl+Space, Ctrl+Space"

    def _run(self, existing: str) -> str | None:
        written: dict[str, str] = {}

        def _capture(_ctx, path, content):
            written["content"] = content

        ctx = _ctx(Path("/tmp/kdictate-runtime"))
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(Path, "read_text", return_value=existing):
                with mock.patch.object(install, "write_home_file", _capture):
                    install.register_global_shortcut(ctx)
        return written.get("content")

    def test_missing_section_is_added(self) -> None:
        result = self._run("[services][other.desktop]\n_launch=Ctrl+X\n")
        self.assertIsNotNone(result)
        self.assertIn(install.TOGGLE_DESKTOP_NAME, result or "")
        self.assertIn(self.ENTRY, result or "")

    def test_a_correct_binding_is_left_untouched(self) -> None:
        existing = f"[services][{install.TOGGLE_DESKTOP_NAME}]\n{self.ENTRY}\n"
        self.assertIsNone(self._run(existing), "rewrote a file that was already correct")

    def test_a_changed_binding_is_restored(self) -> None:
        """The case that made this unrepairable: section present, entry wrong.

        Returning as soon as the section was found meant a Plasma shortcut
        reset left the binding broken with no way to fix it.
        """

        existing = (
            f"[services][{install.TOGGLE_DESKTOP_NAME}]\n"
            "_launch=Ctrl+Alt+Z, Ctrl+Alt+Z\n"
        )
        result = self._run(existing) or ""
        self.assertIn(self.ENTRY, result)
        self.assertNotIn("Ctrl+Alt+Z", result)

    def test_a_removed_entry_is_restored(self) -> None:
        existing = (
            f"[services][{install.TOGGLE_DESKTOP_NAME}]\n"
            "_k_friendly_name=KDictate\n"
            "\n"
            "[services][other.desktop]\n"
            "_launch=Ctrl+X\n"
        )
        result = self._run(existing) or ""
        self.assertIn(self.ENTRY, result)
        # Restored inside our own section, not appended to the other one.
        ours = result.split(f"[services][{install.TOGGLE_DESKTOP_NAME}]")[1]
        self.assertIn(self.ENTRY, ours.split("[services][other.desktop]")[0])
        self.assertIn("_launch=Ctrl+X", result)


class StaleEngineProcessTests(unittest.TestCase):
    """Upgrading replaces the engine on disk; running copies keep the old code."""

    def setUp(self) -> None:
        self.enterContext(
            mock.patch.object(install.shutil, "which", return_value="/usr/bin/pgrep"))
        # install_log appends to the real ~/.local/state; keep tests hermetic.
        self.enterContext(mock.patch.object(install, "install_log"))

    @staticmethod
    def _pgrep(stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["pgrep"], returncode=0 if stdout.strip() else 1,
            stdout=stdout, stderr="")

    def test_nothing_running_signals_nothing(self) -> None:
        with mock.patch.object(
            install, "run_command", return_value=self._pgrep("")
        ) as run:
            self.assertEqual(install.clear_stale_engine_processes(), 0)
        # Only the probe ran; no kill was issued.
        self.assertEqual(len(run.call_args_list), 1)
        self.assertEqual(run.call_args_list[0].args[0][0], "pgrep")

    def test_running_engines_are_terminated_and_counted(self) -> None:
        side_effect = [
            self._pgrep("1975\n757291\n"),   # initial probe: two engines
            self._pgrep(""),                 # pkill
            self._pgrep(""),                 # survivor probe: none left
        ]
        with mock.patch.object(install, "run_command", side_effect=side_effect) as run:
            self.assertEqual(install.clear_stale_engine_processes(), 2)
        self.assertEqual(run.call_args_list[1].args[0][0], "pkill")

    def test_survivors_are_force_killed(self) -> None:
        side_effect = [
            self._pgrep("1975\n"),
            self._pgrep(""),
            self._pgrep("1975\n"),   # ignored SIGTERM
            self._pgrep(""),
        ]
        with mock.patch.object(install, "run_command", side_effect=side_effect) as run:
            install.clear_stale_engine_processes()
        self.assertIn("-9", run.call_args_list[-1].args[0])

    def test_selection_is_scoped_to_this_user_and_an_executed_path(self) -> None:
        """The kill must not be able to reach anything but our own engines.

        Matching a bare name would also hit unrelated processes that merely
        mention it; not scoping by uid would reach other users' engines.
        """

        with mock.patch.object(
            install, "run_command", return_value=self._pgrep("")
        ) as run:
            install.clear_stale_engine_processes()

        argv = run.call_args_list[0].args[0]
        self.assertIn("-u", argv)
        self.assertIn(str(os.getuid()), argv)
        self.assertIn("/ibus-engine-kdictate", argv[-1])

    def test_missing_pgrep_is_not_fatal(self) -> None:
        with mock.patch.object(install.shutil, "which", return_value=None):
            with mock.patch.object(install, "run_command") as run:
                self.assertEqual(install.clear_stale_engine_processes(), 0)
        run.assert_not_called()


def _make_ctx(home: Path) -> install.InstallContext:
    return install.InstallContext(
        script_path=home / "install.py", script_dir=home,
        home=home, runtime_dir=home / ".local/share/kdictate")


class InstallLogTests(unittest.TestCase):
    def test_appends_timestamped_line(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "state" / "install.log"
            with mock.patch.object(install, "INSTALL_LOG_PATH", target):
                install.install_log("first")
                install.install_log("second")
            lines = target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith(" first"))
        self.assertTrue(lines[1].endswith(" second"))
        # "YYYY-MM-DD HH:MM:SS message"
        self.assertRegex(lines[0], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")

    def test_logging_failure_is_swallowed(self) -> None:
        unwritable = Path("/proc/definitely/not/writable/install.log")
        with mock.patch.object(install, "INSTALL_LOG_PATH", unwritable):
            install.install_log("must not raise")


class RefreshIbusRegistryTests(unittest.TestCase):
    """The installer must never sever a working input-method stack.

    Killing ibus-daemon mid-session severed KWin's Wayland IM bridge on
    every 0.14-0.16 install (only a relogin recovered typing), so the
    healthy path is now write-cache + engine clearing and nothing else.
    """

    def setUp(self) -> None:
        self.enterContext(mock.patch.object(install, "install_log"))
        self.run_command = self.enterContext(
            mock.patch.object(install, "run_command", return_value=_completed()))
        self.clear = self.enterContext(
            mock.patch.object(install, "clear_stale_engine_processes", return_value=0))
        self.repair = self.enterContext(
            mock.patch.object(install, "repair_wayland_im_bridge"))
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ctx = _make_ctx(Path(tmp.name))

    def _commands(self) -> list[list[str]]:
        return [list(call.args[0]) for call in self.run_command.call_args_list]

    def test_healthy_stack_is_left_untouched(self) -> None:
        with mock.patch.object(install, "_kwin_wayland_running", return_value=True), \
             mock.patch.object(install, "_wayland_bridge_pids", return_value=["100"]), \
             mock.patch.object(install, "_ibus_daemon_pids", return_value=["200"]):
            install.refresh_ibus_registry(self.ctx)

        self.repair.assert_not_called()
        self.clear.assert_called_once()
        for argv in self._commands():
            self.assertNotIn("pkill", argv[0], f"healthy path must not kill: {argv}")
            self.assertNotEqual(argv[0], "ibus-daemon")
        # write-cache still refreshes the registry.
        self.assertTrue(any(argv[:2] == ["ibus", "write-cache"] for argv in self._commands()))

    def test_missing_bridge_triggers_repair(self) -> None:
        with mock.patch.object(install, "_kwin_wayland_running", return_value=True), \
             mock.patch.object(install, "_wayland_bridge_pids", return_value=[]), \
             mock.patch.object(install, "_ibus_daemon_pids", return_value=["200"]):
            install.refresh_ibus_registry(self.ctx)
        self.repair.assert_called_once()

    def test_non_kde_session_without_daemon_starts_plain_daemon(self) -> None:
        with mock.patch.object(install, "_kwin_wayland_running", return_value=False), \
             mock.patch.object(install, "_ibus_daemon_pids", return_value=[]):
            install.refresh_ibus_registry(self.ctx)
        self.repair.assert_not_called()
        self.assertTrue(any(argv[0] == "ibus-daemon" for argv in self._commands()))

    def test_non_kde_session_with_daemon_is_untouched(self) -> None:
        with mock.patch.object(install, "_kwin_wayland_running", return_value=False), \
             mock.patch.object(install, "_ibus_daemon_pids", return_value=["200"]):
            install.refresh_ibus_registry(self.ctx)
        self.repair.assert_not_called()
        self.assertFalse(any(argv[0] == "ibus-daemon" for argv in self._commands()))

    def test_missing_pgrep_means_unknown_state_and_hands_off(self) -> None:
        """No probes → no verdict. A healthy session must not be misread as a
        cold start: ``ibus-daemon -r`` (--replace) would evict the
        compositor-managed daemon and sever the stack."""

        with mock.patch.object(install.shutil, "which",
                               side_effect=lambda name: None if name == "pgrep" else f"/usr/bin/{name}"):
            install.refresh_ibus_registry(self.ctx)
        self.repair.assert_not_called()
        for argv in self._commands():
            self.assertNotEqual(argv[0], "ibus-daemon")
            self.assertNotEqual(argv[0], "pkill")


class RepairWaylandImBridgeTests(unittest.TestCase):
    """KWin relaunch of the IM stack via the kwinrc InputMethod flip."""

    def setUp(self) -> None:
        self.install_log = self.enterContext(mock.patch.object(install, "install_log"))
        self.enterContext(mock.patch.object(install.time, "sleep"))
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.ctx = _make_ctx(self.home)
        # A real file so KDE_VIRTUAL_KEYBOARD_DESKTOP.is_file() passes.
        desktop = self.home / "org.freedesktop.IBus.Panel.Wayland.Gtk3.desktop"
        desktop.write_text("[Desktop Entry]\n", encoding="utf-8")
        self.enterContext(
            mock.patch.object(install, "KDE_VIRTUAL_KEYBOARD_DESKTOP", desktop))
        self.enterContext(
            mock.patch.object(install.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"))

    def _logged(self) -> str:
        return " | ".join(str(call.args[0]) for call in self.install_log.call_args_list)

    def test_missing_tools_abort_before_any_mutation(self) -> None:
        with mock.patch.object(install.shutil, "which", return_value=None), \
             mock.patch.object(install, "run_command") as run:
            install.repair_wayland_im_bridge(self.ctx)
        run.assert_not_called()
        self.assertIn("skipped", self._logged())

    def test_unreachable_kwin_aborts_before_killing_daemon(self) -> None:
        with mock.patch.object(
            install, "run_command", return_value=_completed(returncode=1)
        ), mock.patch.object(install, "_ibus_daemon_pids", return_value=["300"]), \
             mock.patch.object(install.os, "kill") as kill:
            install.repair_wayland_im_bridge(self.ctx)
        kill.assert_not_called()
        self.assertIn("not reachable", self._logged())

    def test_probe_reconfigure_healing_the_stack_skips_the_flip(self) -> None:
        """First-install race: the probe reconfigure may itself launch the bridge.

        configure_kwin_input_method has just written InputMethod, so KWin's
        first sight of it is our reachability probe — which can spawn the
        stack. Killing that just-healthy daemon would stake a working session
        on the flip succeeding.
        """

        run = self.enterContext(
            mock.patch.object(install, "run_command", return_value=_completed()))
        with mock.patch.object(install, "_ibus_daemon_pids", return_value=["300"]), \
             mock.patch.object(install, "_wayland_bridge_pids", return_value=["400"]):
            install.repair_wayland_im_bridge(self.ctx)

        commands = [list(call.args[0]) for call in run.call_args_list]
        self.assertFalse(any(argv[0] == "pkill" for argv in commands))
        self.assertFalse(any(argv[0] == "kwriteconfig6" for argv in commands))
        self.assertIn("became healthy", self._logged())

    def test_interrupt_between_delete_and_restore_still_restores(self) -> None:
        """A Ctrl+C mid-flip must not leave kwinrc without an InputMethod key.

        Without the key, KWin never launches a bridge again — even at the
        next login — and the promised relogin recovery is gone.
        """

        run = self.enterContext(
            mock.patch.object(install, "run_command", return_value=_completed()))
        # sleep #1 is the post-probe recheck; sleep #2 sits between the
        # kwinrc delete and the restore — interrupt there.
        sleeps = iter([None, KeyboardInterrupt()])

        def _sleep(_seconds: float) -> None:
            outcome = next(sleeps, None)
            if isinstance(outcome, BaseException):
                raise outcome

        with mock.patch.object(install.time, "sleep", side_effect=_sleep), \
             mock.patch.object(install, "_ibus_daemon_pids", return_value=[]), \
             mock.patch.object(install, "_wayland_bridge_pids", return_value=[]), \
             self.assertRaises(KeyboardInterrupt):
            install.repair_wayland_im_bridge(self.ctx)

        commands = [list(map(str, call.args[0])) for call in run.call_args_list]
        restores = [argv for argv in commands
                    if argv[0] == "kwriteconfig6"
                    and str(install.KDE_VIRTUAL_KEYBOARD_DESKTOP) in argv]
        self.assertEqual(len(restores), 1)

    def test_flip_sequence_and_success(self) -> None:
        run = self.enterContext(
            mock.patch.object(install, "run_command", return_value=_completed()))
        kill = self.enterContext(mock.patch.object(install.os, "kill"))
        # Recheck after the probe still sees no bridge; the post-flip poll does.
        with mock.patch.object(install, "_ibus_daemon_pids", return_value=["300"]), \
             mock.patch.object(install, "_wayland_bridge_pids",
                               side_effect=[[], ["400"]]):
            install.repair_wayland_im_bridge(self.ctx)

        commands = [list(map(str, call.args[0])) for call in run.call_args_list]
        # The bridgeless daemon is cleared by its session-filtered PID —
        # never a UID-wide pkill that could reach other sessions' daemons.
        kill.assert_called_once_with(300, install.signal.SIGTERM)
        self.assertFalse(any(argv[0] == "pkill" and "ibus-daemon" in argv for argv in commands))
        # InputMethod is deleted, then restored — the change is what makes
        # KWin relaunch the stack on reconfigure.
        deletes = [argv for argv in commands
                   if argv[0] == "kwriteconfig6" and "--delete" in argv]
        restores = [argv for argv in commands
                    if argv[0] == "kwriteconfig6" and str(install.KDE_VIRTUAL_KEYBOARD_DESKTOP) in argv]
        self.assertEqual(len(deletes), 1)
        self.assertEqual(len(restores), 1)
        self.assertLess(commands.index(deletes[0]), commands.index(restores[0]))
        reconfigures = [argv for argv in commands if "reconfigure" in argv]
        self.assertGreaterEqual(len(reconfigures), 2)
        self.assertIn("succeeded", self._logged())

    def test_bridge_never_appearing_starts_fallback_daemon(self) -> None:
        run = self.enterContext(
            mock.patch.object(install, "run_command", return_value=_completed()))
        with mock.patch.object(install, "_ibus_daemon_pids", return_value=[]), \
             mock.patch.object(install, "_wayland_bridge_pids", return_value=[]):
            install.repair_wayland_im_bridge(self.ctx)
        commands = [list(call.args[0]) for call in run.call_args_list]
        self.assertTrue(any(argv[0] == "ibus-daemon" for argv in commands))
        self.assertIn("FAILED", self._logged())


class SessionScopingTests(unittest.TestCase):
    """UID-wide pgrep must not mix up concurrent graphical sessions."""

    def _filter(
        self, pids: list[str], environs: dict[str, bytes],
        display: str | None = "wayland-0",
    ) -> list[str]:
        env = {"WAYLAND_DISPLAY": display} if display else {}

        def fake_path(spec: str) -> mock.Mock:
            proc = mock.Mock()
            data = environs.get(spec)
            if data is None:
                proc.read_bytes.side_effect = OSError("gone")
            else:
                proc.read_bytes.return_value = data
            return proc

        with mock.patch.dict(install.os.environ, env, clear=True), \
             mock.patch.object(install, "Path", side_effect=fake_path):
            return install._pids_in_current_session(pids)

    def test_without_display_env_everything_is_kept(self) -> None:
        self.assertEqual(self._filter(["1", "2"], {}, display=None), ["1", "2"])

    def test_only_this_sessions_processes_survive(self) -> None:
        environs = {
            "/proc/1/environ": b"WAYLAND_DISPLAY=wayland-0\0HOME=/home/x\0",
            "/proc/2/environ": b"WAYLAND_DISPLAY=wayland-1\0HOME=/home/x\0",
            "/proc/3/environ": b"HOME=/home/x\0",  # no display var: keep
            # /proc/4 vanished between pgrep and the read: drop
        }
        self.assertEqual(
            self._filter(["1", "2", "3", "4"], environs), ["1", "3"])

    def test_display_only_process_matches_a_wayland_session(self) -> None:
        """Each display variable matches its own value, not each other's.

        A KDE Wayland session exports both WAYLAND_DISPLAY=wayland-0 and
        DISPLAY=:0; an XWayland-facing process carrying only DISPLAY=:0 is
        still ours and must not be filtered out.
        """

        environs = {"/proc/1/environ": b"DISPLAY=:0\0HOME=/home/x\0"}
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}

        def fake_path(spec: str) -> mock.Mock:
            proc = mock.Mock()
            proc.read_bytes.return_value = environs[spec]
            return proc

        with mock.patch.dict(install.os.environ, env, clear=True), \
             mock.patch.object(install, "Path", side_effect=fake_path):
            self.assertEqual(install._pids_in_current_session(["1"]), ["1"])


class CheckImStackTests(unittest.TestCase):
    """The self-check must catch a green-checklist install over a dead stack."""

    def _run_check(
        self, *,
        daemons: list[str], kwin: bool, bridge: list[str], engines: list[str],
        active_engine: str = DBUS_INTERFACE, name_has_owner: str = "(true,)",
    ) -> list[str]:
        def fake_run(
            command: list[str | Path], **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            del kwargs
            argv = [str(part) for part in command]
            if argv[:2] == ["ibus", "engine"]:
                return _completed(stdout=f"{active_engine}\n")
            if argv[0] == "gdbus":
                return _completed(stdout=f"{name_has_owner}\n")
            raise AssertionError(f"unexpected command {argv}")

        with mock.patch.object(install, "_ibus_daemon_pids", return_value=daemons), \
             mock.patch.object(install, "_kwin_wayland_running", return_value=kwin), \
             mock.patch.object(install, "_wayland_bridge_pids", return_value=bridge), \
             mock.patch.object(install, "_pgrep_pids", return_value=engines), \
             mock.patch.object(install, "run_command", side_effect=fake_run), \
             mock.patch.object(install.shutil, "which", return_value="/usr/bin/gdbus"):
            return install.check_im_stack()

    def test_healthy_stack_reports_no_problems(self) -> None:
        problems = self._run_check(
            daemons=["1"], kwin=True, bridge=["2"], engines=["3"])
        self.assertEqual(problems, [])

    def test_missing_bridge_is_reported_with_relogin_hint(self) -> None:
        problems = self._run_check(
            daemons=["1"], kwin=True, bridge=[], engines=["3"])
        self.assertEqual(len(problems), 1)
        self.assertIn("log out", problems[0])

    def test_bridge_not_required_without_kwin_wayland(self) -> None:
        problems = self._run_check(
            daemons=["1"], kwin=False, bridge=[], engines=["3"])
        self.assertEqual(problems, [])

    def test_wrong_active_engine_is_reported(self) -> None:
        problems = self._run_check(
            daemons=["1"], kwin=True, bridge=["2"], engines=["3"],
            active_engine="xkb:us::eng")
        self.assertTrue(any("active IBus engine" in p for p in problems))

    def test_unowned_daemon_bus_name_is_reported(self) -> None:
        problems = self._run_check(
            daemons=["1"], kwin=True, bridge=["2"], engines=["3"],
            name_has_owner="(false,)")
        self.assertTrue(any("does not own" in p for p in problems))

    def test_dead_stack_reports_every_layer(self) -> None:
        problems = self._run_check(
            daemons=[], kwin=True, bridge=[], engines=[],
            active_engine="", name_has_owner="(false,)")
        self.assertGreaterEqual(len(problems), 4)


class InstallerArgumentTests(unittest.TestCase):
    def test_reconfigure_flag_exists_and_defaults_off(self) -> None:
        """Repairing an install at the current version must be possible.

        Every configuration step is idempotent and re-running them is the
        documented repair path. The version gate skips all of them when the
        versions match, so without an override a user whose install broke *at
        the current version* could only fix it by editing app_metadata.py.
        """

        self.assertFalse(install.parse_args([]).reconfigure)
        self.assertTrue(install.parse_args(["--reconfigure"]).reconfigure)


class PromptUpdateTests(unittest.TestCase):
    def test_empty_answer_defaults_to_updating(self) -> None:
        with mock.patch("builtins.input", return_value=""):
            self.assertTrue(install._prompt_update("0.13.0"))

    def test_declining_returns_false(self) -> None:
        with mock.patch("builtins.input", return_value="n"):
            self.assertFalse(install._prompt_update("0.13.0"))

    def test_fresh_install_proceeds_without_prompting(self) -> None:
        with mock.patch("builtins.input", side_effect=AssertionError("prompted")):
            self.assertTrue(install._prompt_update(None))


class QuietFailureTests(unittest.TestCase):
    """Suppressed output is only safe if failures still say what went wrong."""

    def test_failure_surfaces_the_tail_of_the_captured_output(self) -> None:
        result = subprocess.CompletedProcess(
            args=["makepkg"],
            returncode=2,
            stdout="\n".join(f"line {n}" for n in range(1, 41)),
            stderr="ERROR: the actual cause",
        )
        with self.assertRaises(SystemExit):
            with mock.patch("sys.stderr", new=io.StringIO()) as err:
                install._die_with_output("Package build failed", result, tail=5)
        printed = err.getvalue()
        self.assertIn("Package build failed", printed)
        self.assertIn("exit 2", printed)
        self.assertIn("ERROR: the actual cause", printed)
        # Tail only -- the whole log would defeat the point of running quiet.
        self.assertNotIn("line 1\n", printed)

    def test_missing_build_dependencies_are_reported_by_package_name(self) -> None:
        failed = subprocess.CompletedProcess(args=["python"], returncode=1, stdout="", stderr="")
        with mock.patch.object(install, "run_command", return_value=failed):
            self.assertEqual(
                install._require_build_dependencies(),
                ["python-build", "python-installer"],
            )

    def test_present_build_dependencies_report_nothing_missing(self) -> None:
        ok = subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr="")
        with mock.patch.object(install, "run_command", return_value=ok):
            self.assertEqual(install._require_build_dependencies(), [])


class EnsureBuildDependenciesTests(unittest.TestCase):
    """The installer installs what the rebuild needs, rather than punting."""

    def _patch_env(self, missing: list[str]):
        self.enterContext(mock.patch.object(
            install, "_require_build_dependencies", side_effect=[missing, []]))
        self.enterContext(mock.patch.object(install, "_detect_distro", return_value="arch"))
        self.enterContext(mock.patch.object(install.shutil, "which", return_value="/usr/bin/sudo"))

    def test_accepting_installs_the_packages_as_dependencies(self) -> None:
        """Build tooling is installed --asdeps, because that is what it is.

        Marking it explicit would stop an orphan sweep ever reclaiming it,
        which is the installer overriding the machine's cleanup policy to
        spare itself a round trip. Whether build tooling stays installed is
        the user's call; all this has to guarantee is that a sweep never
        leaves the next rebuild dead-ended, which re-installing on demand
        already does.
        """

        self._patch_env(["python-build", "python-installer"])
        ok = subprocess.CompletedProcess(args=["pacman"], returncode=0, stdout="", stderr="")
        with mock.patch("builtins.input", return_value=""):
            with mock.patch.object(install, "run_command", return_value=ok) as run:
                install.ensure_build_dependencies()

        argv = run.call_args.args[0]
        self.assertEqual(argv[:5], ["sudo", "pacman", "-S", "--needed", "--asdeps"])
        self.assertIn("python-build", argv)
        self.assertIn("python-installer", argv)

    def test_declining_stops_with_the_manual_command(self) -> None:
        self._patch_env(["python-installer"])
        with mock.patch("builtins.input", return_value="n"):
            with self.assertRaises(SystemExit):
                with mock.patch("sys.stderr", new=io.StringIO()) as err:
                    install.ensure_build_dependencies()
        self.assertIn("python-installer", err.getvalue())

    def test_nothing_missing_prompts_for_nothing(self) -> None:
        with mock.patch.object(install, "_require_build_dependencies", return_value=[]):
            with mock.patch("builtins.input", side_effect=AssertionError("prompted")):
                install.ensure_build_dependencies()
