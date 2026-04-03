"""Tests for the installed packaging and service assets."""

from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from whisper_dictate.constants import DBUS_INTERFACE
from whisper_dictate.ibus_engine.engine import ENGINE_NAME


class PackagingAssetTests(unittest.TestCase):
    """Validate that the IBus packaging metadata stays internally consistent."""

    def test_component_metadata_matches_root_identity(self) -> None:
        component_path = Path("packaging/io.github.pizzimenti.WhisperDictate.component.xml")
        root = ET.fromstring(component_path.read_text(encoding="utf-8"))

        self.assertEqual(root.findtext("name"), "io.github.pizzimenti.WhisperDictate")
        self.assertEqual(root.findtext("exec"), "@@ENGINE_EXEC@@")

        engine = root.find("engine")
        self.assertIsNotNone(engine)
        assert engine is not None
        self.assertEqual(engine.findtext("name"), DBUS_INTERFACE)
        self.assertEqual(ENGINE_NAME, DBUS_INTERFACE)
        self.assertEqual(engine.findtext("longname"), "Whisper Dictate")
        self.assertEqual(engine.findtext("language"), "en")

    def test_dbus_and_systemd_service_files_reference_the_frozen_identity(self) -> None:
        dbus_service_path = Path("packaging/io.github.pizzimenti.WhisperDictate.service")
        systemd_service_path = Path("systemd/io.github.pizzimenti.WhisperDictate.service")
        launcher_template_path = Path("packaging/ibus-engine-whisper-dictate")

        dbus_service = dbus_service_path.read_text(encoding="utf-8")
        systemd_service = systemd_service_path.read_text(encoding="utf-8")
        launcher_template = launcher_template_path.read_text(encoding="utf-8")

        self.assertIn("Name=io.github.pizzimenti.WhisperDictate1", dbus_service)
        self.assertIn("Exec=", dbus_service)
        self.assertIn("io.github.pizzimenti.WhisperDictate.service", systemd_service_path.name)
        self.assertIn("ExecStart=", systemd_service)
        self.assertIn("dictate.py", dbus_service)
        self.assertIn("dictate.py", systemd_service)
        self.assertNotIn("--no-type-output", dbus_service)
        self.assertNotIn("--no-type-output", systemd_service)
        self.assertTrue(launcher_template_path.exists())
        self.assertIn("@@REPO_DIR@@/ibus_engine.py", launcher_template)

    def test_regression_shell_check_is_executable_and_scoped(self) -> None:
        script_path = Path("scripts/check-ibus-only.sh")
        self.assertTrue(script_path.exists())
        script = script_path.read_text(encoding="utf-8")

        self.assertTrue(script.startswith("#!/usr/bin/env bash"))
        self.assertIn("systemd/**", script)
        self.assertIn("packaging/**", script)
        self.assertIn("ibus_engine.py", script)
        self.assertIn("ydotool|dotool|wtype|wl-copy|xdotool|type_text", script)
