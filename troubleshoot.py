#!/usr/bin/env python3
"""Diagnose kdictate configuration on the current system.

Checks every layer of the stack — KWin input method, IBus daemon, engine
registration, systemd service, audio device — and prints a pass/fail line
for each item so problems are visible at a glance.

Run as the normal user (not root):

    python3 troubleshoot.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
RUNTIME_DIR = HOME / ".local" / "share" / "kdictate"
VENV_BIN = RUNTIME_DIR / ".venv" / "bin"
ENGINE_EXEC = VENV_BIN / "ibus-engine-kdictate"
COMPONENT_XML = HOME / ".local" / "share" / "ibus" / "component" / "io.github.pizzimenti.KDictate.component.xml"
IBUS_ENV_FILE = HOME / ".config" / "environment.d" / "60-kdictate-ibus.conf"
PLASMA_ENV_SCRIPT = HOME / ".config" / "plasma-workspace" / "env" / "kdictate-plasma-wayland.sh"
KWINRC = HOME / ".config" / "kwinrc"
SERVICE_NAME = "io.github.pizzimenti.KDictate.service"
DBUS_INTERFACE = "io.github.pizzimenti.KDictate1"
KDE_VIRTUAL_KEYBOARD_DESKTOP = Path(
    "/usr/share/applications/org.freedesktop.IBus.Panel.Wayland.Gtk3.desktop"
)

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_WARN = "\033[33mWARN\033[0m"
_INFO = "\033[36mINFO\033[0m"

_issues: list[str] = []


def _run(*args: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def check(label: str, passed: bool, detail: str = "", warn_only: bool = False) -> bool:
    tag = _PASS if passed else (_WARN if warn_only else _FAIL)
    suffix = f"  {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    if not passed and not warn_only:
        _issues.append(label)
    return passed


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")


# ---------------------------------------------------------------------------
# System binaries
# ---------------------------------------------------------------------------

section("System binaries")
check("ibus", shutil.which("ibus") is not None)
check("ibus-daemon", shutil.which("ibus-daemon") is not None)
check("ibus-ui-gtk3", Path("/usr/lib/ibus/ibus-ui-gtk3").is_file())
check("KDE InputMethod desktop file", KDE_VIRTUAL_KEYBOARD_DESKTOP.is_file(),
      str(KDE_VIRTUAL_KEYBOARD_DESKTOP))

# ---------------------------------------------------------------------------
# Installed files
# ---------------------------------------------------------------------------

section("Installed files")
check("Runtime directory", RUNTIME_DIR.is_dir(), str(RUNTIME_DIR))
check("Python venv", VENV_BIN.is_dir(), str(VENV_BIN))
check("ibus-engine-kdictate binary", ENGINE_EXEC.is_file(), str(ENGINE_EXEC))
check("IBus component XML", COMPONENT_XML.is_file(), str(COMPONENT_XML))
check("environment.d IBus config", IBUS_ENV_FILE.is_file(), str(IBUS_ENV_FILE))
check("Plasma env script", PLASMA_ENV_SCRIPT.is_file(), str(PLASMA_ENV_SCRIPT),
      warn_only=True)

# ---------------------------------------------------------------------------
# environment.d contents
# ---------------------------------------------------------------------------

section("environment.d (60-kdictate-ibus.conf)")
if IBUS_ENV_FILE.is_file():
    env_text = IBUS_ENV_FILE.read_text(encoding="utf-8")
    check("IBUS_COMPONENT_PATH set", "IBUS_COMPONENT_PATH" in env_text)
    check("XMODIFIERS=@im=ibus", "XMODIFIERS=@im=ibus" in env_text)
else:
    check("IBUS_COMPONENT_PATH set", False, "file missing")
    check("XMODIFIERS=@im=ibus", False, "file missing")

# ---------------------------------------------------------------------------
# kwinrc / KWin virtual keyboard
# ---------------------------------------------------------------------------

section("KWin virtual keyboard (kwinrc)")
if KWINRC.is_file():
    kwinrc_text = KWINRC.read_text(encoding="utf-8")
    # KDE may write the key as "InputMethod=" or "InputMethod[$e]=" (the
    # latter tells kconfig to expand $VARs in the value at read time).
    has_im = ("InputMethod=" in kwinrc_text or "InputMethod[$e]=" in kwinrc_text) \
        and "IBus" in kwinrc_text
    has_enabled = "VirtualKeyboardEnabled=true" in kwinrc_text
    check("InputMethod=…IBus…Wayland…desktop", has_im)
    check("VirtualKeyboardEnabled=true", has_enabled)
else:
    check("kwinrc exists", False, str(KWINRC))

# Live KWin D-Bus state
for qdbus in ("qdbus6", "qdbus"):
    if shutil.which(qdbus) is None:
        continue
    try:
        avail = _run(qdbus, "org.kde.KWin", "/VirtualKeyboard",
                     "org.kde.kwin.VirtualKeyboard.available").stdout.strip()
        enabled = _run(qdbus, "org.kde.KWin", "/VirtualKeyboard",
                       "org.kde.kwin.VirtualKeyboard.enabled").stdout.strip()
        check("KWin VirtualKeyboard.available (live)", avail == "true",
              f"(value={avail!r})")
        check("KWin VirtualKeyboard.enabled (live)", enabled == "true",
              f"(value={enabled!r})")
    except Exception as exc:  # noqa: BLE001
        check("KWin VirtualKeyboard D-Bus query", False, str(exc))
    break

# ---------------------------------------------------------------------------
# IBus daemon and engine
# ---------------------------------------------------------------------------

section("IBus daemon and engine")

ibus_procs = _run("pgrep", "-a", "ibus").stdout.strip()
daemon_running = "ibus-daemon" in ibus_procs
check("ibus-daemon process", daemon_running)
wayland_ui_running = "ibus-ui-gtk3" in ibus_procs and "--enable-wayland-im" in ibus_procs
check("ibus-ui-gtk3 --enable-wayland-im process", wayland_ui_running,
      "(required for KDE Wayland text insertion)")
engine_running = "ibus-engine-kdictate" in ibus_procs
check("ibus-engine-kdictate process", engine_running)

if daemon_running:
    try:
        result = _run("ibus", "engine")
        active_engine = result.stdout.strip()
        check("Active IBus engine is KDictate",
              active_engine == DBUS_INTERFACE,
              f"(active={active_engine!r})", warn_only=not engine_running)
    except Exception as exc:  # noqa: BLE001
        check("ibus engine query", False, str(exc))

try:
    result = _run("dconf", "read", "/desktop/ibus/general/preload-engines")
    preload = result.stdout.strip()
    check("KDictate in preload-engines",
          DBUS_INTERFACE in preload,
          f"(value={preload!r})")
except Exception as exc:  # noqa: BLE001
    check("dconf preload-engines query", False, str(exc), warn_only=True)

# ---------------------------------------------------------------------------
# systemd service
# ---------------------------------------------------------------------------

section("systemd user service")
try:
    result = _run("systemctl", "--user", "is-active", SERVICE_NAME)
    active = result.stdout.strip() == "active"
    check(f"{SERVICE_NAME} is active", active, result.stdout.strip())
    result2 = _run("systemctl", "--user", "is-enabled", SERVICE_NAME)
    check(f"{SERVICE_NAME} is enabled", result2.stdout.strip() == "enabled",
          result2.stdout.strip())
except Exception as exc:  # noqa: BLE001
    check("systemctl query", False, str(exc))

# D-Bus ping
try:
    ping = _run("gdbus", "call", "--session",
                "--dest", DBUS_INTERFACE,
                "--object-path", f"/{DBUS_INTERFACE.replace('.', '/')}",
                "--method", f"{DBUS_INTERFACE}.Ping")
    check("Daemon D-Bus Ping", ping.returncode == 0,
          ping.stdout.strip() or ping.stderr.strip())
except Exception as exc:  # noqa: BLE001
    check("Daemon D-Bus Ping", False, str(exc))

# ---------------------------------------------------------------------------
# Audio device
# ---------------------------------------------------------------------------

section("Audio input device")
try:
    result = _run("pactl", "get-default-source")
    source = result.stdout.strip()
    is_monitor = source.endswith(".monitor")
    check("Default source is not a monitor", not is_monitor, f"(source={source!r})",
          warn_only=is_monitor)
    if source:
        desc_result = _run("pactl", "list", "sources")
        in_target = False
        description = source
        for line in desc_result.stdout.splitlines():
            stripped = line.strip()
            parts = stripped.split(None, 1)
            if stripped.startswith("Name:") and len(parts) > 1 and parts[1] == source:
                in_target = True
            elif in_target and stripped.startswith("Description:"):
                description = stripped.split(":", 1)[1].strip()
                break
        print(f"  [{_INFO}] Default source: {description}")
except Exception as exc:  # noqa: BLE001
    check("pactl query", False, str(exc), warn_only=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if _issues:
    print(f"  \033[31m{len(_issues)} issue(s) found:\033[0m")
    for issue in _issues:
        print(f"    • {issue}")
    print()
    qdbus_bin = next(
        (b for b in ("qdbus6", "qdbus") if shutil.which(b) is not None),
        None,
    )
    if qdbus_bin is None:
        print(
            "  Hot-start needs qdbus6 (or qdbus) on PATH, but neither was found.\n"
            "  Install one and re-run, or log out and back in to apply the new\n"
            "  InputMethod setting."
        )
    else:
        print("  Hot-start IBus for the current session (no logout needed):")
        print(f"    {qdbus_bin} org.kde.KWin /KWin reconfigure && \\")
        print("      pkill -x ibus-daemon; sleep 0.5 && \\")
        print(f"      {qdbus_bin} --literal org.kde.KWin /VirtualKeyboard \\")
        print("        org.freedesktop.DBus.Properties.Set \\")
        print("        org.kde.kwin.VirtualKeyboard enabled false && sleep 0.5 && \\")
        print(f"      {qdbus_bin} --literal org.kde.KWin /VirtualKeyboard \\")
        print("        org.freedesktop.DBus.Properties.Set \\")
        print("        org.kde.kwin.VirtualKeyboard enabled true")
    sys.exit(1)
else:
    print("  \033[32mAll checks passed.\033[0m")
    sys.exit(0)
