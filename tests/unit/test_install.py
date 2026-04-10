"""Tests for the Python installer helpers."""

from __future__ import annotations

import unittest

import install
from kdictate.constants import DBUS_INTERFACE


class InstallHelperTests(unittest.TestCase):
    def test_next_preload_engines_returns_none_when_engine_is_present(self) -> None:
        current = f"['xkb:de::ger', '{DBUS_INTERFACE}']"
        self.assertIsNone(install.next_preload_engines(current, DBUS_INTERFACE))

    def test_next_preload_engines_adds_only_kdictate_for_empty_values(self) -> None:
        self.assertEqual(
            install.next_preload_engines("@as []", DBUS_INTERFACE),
            f"['{DBUS_INTERFACE}']",
        )

    def test_next_preload_engines_strips_typed_prefix_before_append(self) -> None:
        self.assertEqual(
            install.next_preload_engines("@as ['xkb:de::ger']", DBUS_INTERFACE),
            f"['xkb:de::ger', '{DBUS_INTERFACE}']",
        )

    def test_next_preload_engines_rejects_unexpected_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected dconf preload-engines value"):
            install.next_preload_engines("not-a-list", DBUS_INTERFACE)
