"""Tests for the Python installer helpers."""

from __future__ import annotations

import io
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
        self.assertIn("-P", argv)
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
            self.assertIn("-P", call.args[0])
            self.assertNotEqual(call.kwargs.get("cwd"), None)


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

    def test_accepting_installs_the_packages_without_asdeps(self) -> None:
        """--asdeps is what made them vanish, so it must not be used.

        makepkg installs build dependencies as dependencies. Once a build
        finishes nothing depends on them, so they become orphans and the next
        `pacman -Rns $(pacman -Qtdq)` sweep removes them — which is exactly
        how this machine lost them two days after installing them. Rebuilding
        is the normal update path here, so they are wanted explicitly.
        """

        self._patch_env(["python-build", "python-installer"])
        ok = subprocess.CompletedProcess(args=["pacman"], returncode=0, stdout="", stderr="")
        with mock.patch("builtins.input", return_value=""):
            with mock.patch.object(install, "run_command", return_value=ok) as run:
                install.ensure_build_dependencies()

        argv = run.call_args.args[0]
        self.assertEqual(argv[:4], ["sudo", "pacman", "-S", "--needed"])
        self.assertIn("python-build", argv)
        self.assertIn("python-installer", argv)
        self.assertNotIn("--asdeps", argv)

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
