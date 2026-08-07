"""Regression checks for forbidden injector backends in active paths."""

from __future__ import annotations

from pathlib import Path
import unittest


FORBIDDEN_TOKENS = ("ydotool", "dotool", "wtype", "wl-copy", "xdotool")
ALLOWED_SUFFIXES = {".py", ".sh", ".xml", ".service", ".desktop", ".md"}
ACTIVE_PATHS = (
    Path("README.md"),
    Path("install.py"),
    Path("kdictate"),
    Path("packaging"),
)

# makepkg's staging trees, which live under packaging/ and are gitignored.
# They hold extracted whisper.cpp sources and pip-vendored wheels — none of
# it ours, and numpy alone contains "wtype" in several unrelated places. This
# guard exists to catch *our* code regressing to a keyboard-injector backend,
# so scanning vendored third-party source only turns it into a false alarm
# that fires on any machine where a package build has been run.
EXCLUDED_ROOTS = (
    Path("packaging/src"),
    Path("packaging/pkg"),
)


def _is_excluded(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in EXCLUDED_ROOTS)


class ForbiddenPathRegressionTests(unittest.TestCase):
    """Protect the active IBus-focused code paths from injector regressions."""

    def test_forbidden_backends_are_absent_from_active_paths(self) -> None:
        violations: list[str] = []
        for root in ACTIVE_PATHS:
            paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for path in paths:
                if path.suffix not in ALLOWED_SUFFIXES or _is_excluded(path):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for token in FORBIDDEN_TOKENS:
                    if token in text:
                        violations.append(f"{path}: {token}")

        self.assertEqual(violations, [], msg="Forbidden backend references found:\n" + "\n".join(violations))
