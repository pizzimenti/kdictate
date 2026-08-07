"""Tests for the Python installer helpers."""

from __future__ import annotations

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
