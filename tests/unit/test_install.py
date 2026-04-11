"""Tests for the Python installer helpers."""

from __future__ import annotations

import unittest

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
        with self.assertRaisesRegex(ValueError, "Unexpected dconf preload-engines value"):
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
        with self.assertRaisesRegex(ValueError, "Unexpected dconf preload-engines value"):
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
