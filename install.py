#!/usr/bin/env python3
"""Install KDictate into the current user's desktop session.

Runs entirely as the invoking user — no root required.  The only
system-level dependency is ``ibus``, which must be installed via the
distro package manager before running this script.

Everything lives under ``$HOME``:

* ``~/.local/share/kdictate/`` — runtime source tree + venv + models
* ``~/.config/systemd/user/`` — user service unit
* ``~/.local/share/dbus-1/services/`` — D-Bus session activation
* ``~/.local/share/ibus/component/`` — IBus engine metadata
* ``~/.config/environment.d/`` — IBUS_COMPONENT_PATH
* ``~/.config/plasma-workspace/env/`` — Plasma Wayland env hook
* ``~/.local/share/applications/`` — toggle .desktop file
* ``~/.config/kglobalshortcutsrc`` — Ctrl+Space binding
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Final, NoReturn

from kdictate import __version__
from kdictate.app_metadata import (
    DEFAULT_MODEL_HF_REPO,
    DEFAULT_MODEL_NAME,
    GGML_MODEL_FILENAME,
    GGML_MODEL_HF_REPO,
    GGML_MODEL_PATH,
)
from kdictate.constants import APP_ROOT_ID, DBUS_BUS_NAME, DBUS_INTERFACE

PACKAGE_NAME: Final[str] = "kdictate"
SERVICE_NAME = f"{APP_ROOT_ID}.service"
DBUS_SERVICE_NAME = f"{DBUS_INTERFACE}.service"
IBUS_COMPONENT_NAME = f"{APP_ROOT_ID}.component.xml"
TOGGLE_DESKTOP_NAME = f"{APP_ROOT_ID}Toggle.desktop"
IBUS_ENV_FILE_NAME = "60-kdictate-ibus.conf"
PLASMA_ENV_SCRIPT_NAME = "kdictate-plasma-wayland.sh"
KDE_VIRTUAL_KEYBOARD_DESKTOP = Path(
    "/usr/share/applications/org.freedesktop.IBus.Panel.Wayland.Gtk3.desktop"
)


# -------------------------------------------------------------------
# Context
# -------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InstallContext:
    script_path: Path
    script_dir: Path
    home: Path
    runtime_dir: Path
    gpu: bool = False

    @property
    def venv_dir(self) -> Path:
        return self.runtime_dir / ".venv"

    @property
    def python_bin(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def pip_bin(self) -> Path:
        return self.venv_dir / "bin" / "pip"

    @property
    def engine_exec(self) -> Path:
        return self.venv_dir / "bin" / "ibus-engine-kdictate"

    @property
    def replacements(self) -> Mapping[str, str]:
        return {
            "@@REPO_DIR@@": str(self.runtime_dir),
            "@@ENGINE_EXEC@@": str(self.engine_exec),
            "@@HOME@@": str(self.home),
            "@@APP_VERSION@@": __version__,
            "@@BACKEND_FLAGS@@": " --backend gpu" if self.gpu else " --backend cpu",
        }


# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------

_TOTAL_STEPS = 11
_current_step = 0


def log(message: str) -> None:
    print(f"    {message}")


def step(message: str) -> None:
    global _current_step  # noqa: PLW0603
    _current_step += 1
    print(f"  \u2705 [{_current_step}/{_TOTAL_STEPS}] {message}", end="", flush=True)


def step_done(detail: str = "") -> None:
    print(f" ({detail})" if detail else "")


def die(message: str) -> NoReturn:
    print(f"\n  \u274c  {message}\n", file=sys.stderr)
    raise SystemExit(1)


# -------------------------------------------------------------------
# Persistent install log
# -------------------------------------------------------------------

INSTALL_LOG_PATH = Path.home() / ".local" / "state" / "kdictate" / "install.log"


def install_log(message: str) -> None:
    """Append a timestamped line to the persistent install log.

    The checklist UI intentionally hides what the installer does to the
    session's input-method stack, which made past breakage undiagnosable
    after the fact ("what did the install actually kill?"). Every IBus
    mutation \u2014 process kills, daemon state decisions, repair attempts and
    their outcomes \u2014 lands here so the next incident is answerable from
    ``~/.local/state/kdictate/install.log`` alongside the daemon and
    engine logs in the same directory.

    Best-effort by design: a logging failure must never break an install.
    """

    try:
        INSTALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with INSTALL_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {message}\n")
    except OSError:
        pass


# -------------------------------------------------------------------
# Distro detection
# -------------------------------------------------------------------

def _detect_distro() -> str:
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8").lower()
    except FileNotFoundError:
        return "unknown"
    if "arch" in text or "manjaro" in text or "endeavour" in text:
        return "arch"
    if "ubuntu" in text or "debian" in text or "mint" in text:
        return "debian"
    if "fedora" in text or "rhel" in text or "centos" in text:
        return "fedora"
    return "unknown"


def _pkg_hint(distro: str, pkg: str) -> str:
    """Return the distro-appropriate install command for *pkg*."""
    if distro == "arch":
        return f"sudo pacman -S --needed {pkg}"
    if distro == "debian":
        return f"sudo apt install {pkg}"
    if distro == "fedora":
        return f"sudo dnf install {pkg}"
    return f"install {pkg} with your package manager"


# -------------------------------------------------------------------
# GPU detection and prompt
# -------------------------------------------------------------------

def _detect_gpu() -> tuple[str | None, list[str]]:
    """Return (whisper_cpp_binary, reasons_unavailable)."""
    reasons: list[str] = []
    distro = _detect_distro()

    # Mirror kdictate.backend.find_whisper_cpp resolution so the installer's
    # GPU detection agrees with runtime discovery: an explicit
    # $KDICTATE_WHISPER_CLI override (e.g. a packaging/build-whisper.sh build
    # in a source checkout), then the vendored package binary, then PATH.
    override = os.environ.get("KDICTATE_WHISPER_CLI")
    vendored = Path("/usr/lib/kdictate/bin/whisper-cli")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        binary = override
    elif vendored.is_file() and os.access(vendored, os.X_OK):
        binary = str(vendored)
    else:
        binary = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    if binary is None:
        # Packaged installs resolve to the vendored binary above and never
        # reach here. This hint is for source/dev checkouts: build the
        # *pinned* whisper so dev matches what ships — no AUR, no
        # llama.cpp-vulkan, no daily churn.
        hint = "./packaging/build-whisper.sh   (builds the pinned whisper.cpp)"
        reasons.append(f"whisper.cpp not found\n        Install:  {hint}")

    if shutil.which("vulkaninfo") is None:
        reasons.append(
            f"vulkaninfo not found (needed to verify GPU)\n"
            f"        Install:  {_pkg_hint(distro, 'vulkan-tools')}"
        )
    elif binary is not None:
        try:
            r = subprocess.run(["vulkaninfo", "--summary"],
                               capture_output=True, timeout=5)
            if r.returncode != 0:
                reasons.append("vulkaninfo failed — no Vulkan-capable GPU detected")
        except (OSError, subprocess.TimeoutExpired):
            reasons.append("vulkaninfo timed out or crashed")

    return binary, reasons


def _prompt_backend() -> bool:
    """Auto-detect GPU and ask the user.  Returns True for GPU mode."""
    binary, reasons = _detect_gpu()

    try:
        if not reasons:
            print("  GPU acceleration is available:\n")
            print(f"    whisper.cpp: {binary}")
            print("    Vulkan:      supported\n")
            print("    [1] GPU mode  (whisper.cpp + Vulkan, faster)")
            print("    [2] CPU mode  (faster-whisper, no extra deps)\n")
            while True:
                choice = input("  Select [1/2] (default: 1): ").strip()
                if choice in ("", "1"):
                    return True
                if choice == "2":
                    return False
        else:
            print("  GPU acceleration is not available:\n")
            for reason in reasons:
                print(f"    - {reason}")
            print()
            while True:
                choice = input("  Proceed with CPU-only install? [Y/n]: ").strip().lower()
                if choice in ("", "y", "yes"):
                    return False
                if choice in ("n", "no"):
                    die("Install cancelled.")
    except EOFError:
        die("No interactive input available. Run install.py from a terminal.")

    return False


# -------------------------------------------------------------------
# Shell helpers
# -------------------------------------------------------------------

def require_command(name: str) -> None:
    if shutil.which(name) is None:
        distro = _detect_distro()
        die(f"Required command not found: {name}\n\n      {_pkg_hint(distro, name)}")


def run_command(
    command: list[str | Path], *,
    env: Mapping[str, str] | None = None,
    quiet: bool = False, check: bool = True,
    timeout: float | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [str(p) for p in command]
    proc_env = {**os.environ, **(env or {})}
    return subprocess.run(
        args, check=check, encoding="utf-8", errors="replace",
        capture_output=quiet, env=proc_env, timeout=timeout,
        cwd=None if cwd is None else str(cwd),
    )


# -------------------------------------------------------------------
# File helpers
# -------------------------------------------------------------------

def _ensure_under_home(ctx: InstallContext, dest: Path) -> None:
    resolved = dest.resolve(strict=False)
    if not resolved.is_relative_to(ctx.home.resolve()):
        die(f"Refusing to write outside home: {dest} -> {resolved}")


def write_home_file(ctx: InstallContext, dest: Path, text: str, *, mode: int = 0o644) -> None:
    _ensure_under_home(ctx, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    dest.chmod(mode)


def copy_home_file(ctx: InstallContext, src: Path, dest: Path, *, mode: int = 0o644) -> None:
    _ensure_under_home(ctx, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    dest.chmod(mode)


def render_template(src: Path, replacements: Mapping[str, str]) -> str:
    text = src.read_text(encoding="utf-8")
    for needle, replacement in replacements.items():
        text = text.replace(needle, replacement)
    return text


def install_rendered_file(ctx: InstallContext, src: Path, dest: Path, *, mode: int = 0o644) -> None:
    write_home_file(ctx, dest, render_template(src, ctx.replacements), mode=mode)


# -------------------------------------------------------------------
# Install steps
# -------------------------------------------------------------------

def sync_runtime(ctx: InstallContext) -> None:
    ctx.runtime_dir.mkdir(parents=True, exist_ok=True)
    run_command([
        "rsync", "-a", "--delete", "--delete-excluded", "--exclude=__pycache__",
        f"{ctx.script_dir / 'kdictate'}/", f"{ctx.runtime_dir / 'kdictate'}/",
    ])
    copy_home_file(ctx, ctx.script_dir / "requirements.txt", ctx.runtime_dir / "requirements.txt")
    copy_home_file(ctx, ctx.script_dir / "pyproject.toml", ctx.runtime_dir / "pyproject.toml")


def install_python_environment(ctx: InstallContext) -> None:
    run_command(["python3", "-m", "venv", str(ctx.venv_dir)], quiet=True)
    run_command([ctx.pip_bin, "install", "--upgrade", "pip"], quiet=True)
    run_command([ctx.pip_bin, "install", "-r", ctx.runtime_dir / "requirements.txt"], quiet=True)
    run_command([ctx.pip_bin, "install", "--no-deps", "-e", ctx.runtime_dir], quiet=True)


# Minimum expected size per file, in bytes.  Files smaller than this on disk
# almost certainly come from a previous interrupted snapshot_download (network
# drop mid-transfer, disk full, etc.) — re-download rather than persist a
# corrupt model into runtime.  Values are well below the actual sizes
# (model.bin ~1.62 GB, tokenizer.json ~2.7 MB) so they don't false-fail when
# HF re-publishes the model with slightly different sizes.
_CPU_MODEL_REQUIRED_FILES: Final[tuple[tuple[str, int], ...]] = (
    ("model.bin", 1_500_000_000),
    ("config.json", 1_000),
    ("tokenizer.json", 2_000_000),
    ("vocabulary.json", 500_000),
    ("preprocessor_config.json", 100),
)


def _model_files_present(
    model_dir: Path,
    required: Iterable[tuple[str, int]],
) -> bool:
    """Return True iff every required file exists at or above its minimum size.

    A non-zero size alone is not enough: a truncated download leaves a file
    that exists with some content but is unusable.  The minimum size check
    catches that without a network round-trip to HF.
    """
    for name, min_size in required:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size < min_size:
            return False
    return True


def _hf_download_env(ctx: InstallContext) -> tuple[str, dict[str, str] | None]:
    """(python executable, env) for running huggingface_hub downloads.

    Source installs use the venv python (which has the deps). A packaged
    install has no venv, so use the system python with the package's
    vendored deps on PYTHONPATH.
    """
    if _is_packaged_install():
        vendor = "/usr/lib/kdictate/vendor"
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = vendor + os.pathsep + existing if existing else vendor
        return "/usr/bin/python", env
    return str(ctx.python_bin), None


def download_cpu_model(ctx: InstallContext) -> None:
    """Download the CTranslate2 model with a single clean progress bar.

    Skips the network call entirely if the model is already on disk.
    snapshot_download otherwise issues a HEAD/etag request for every file
    even when nothing needs to download, and a single hung request to
    huggingface.co (silent connection drop) blocks the install indefinitely.
    To force a re-download, delete the model directory.

    Uses ``max_workers=1`` so tqdm shows one bar at a time instead of
    overlapping parallel fetches.
    """
    model_dir = ctx.runtime_dir / DEFAULT_MODEL_NAME
    if _model_files_present(model_dir, _CPU_MODEL_REQUIRED_FILES):
        print(f"  (model already present at {model_dir}, skipping download)")
        return
    py, env = _hf_download_env(ctx)
    subprocess.run([
        py, "-u", "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={DEFAULT_MODEL_HF_REPO!r}, "
        f"local_dir={str(model_dir)!r}, "
        f"max_workers=1)",
    ], check=True, env=env)


def download_gpu_model(ctx: InstallContext) -> None:
    """Download the GGML Q8_0 model (single file, single progress bar).

    Skips the download if the file is already on disk — see
    ``download_cpu_model`` for why this matters.
    """
    GGML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Per-file minimum size, conservatively below the published ~834 MB.  The
    # GGML Q8_0 large-v3-turbo file is single-blob; a partial download leaves a
    # smaller-but-nonzero file that loads as a corrupted model at runtime.
    # See download_cpu_model for the same pattern on the CT2 model.
    if GGML_MODEL_PATH.is_file() and GGML_MODEL_PATH.stat().st_size >= 700_000_000:
        print(f"  (model already present at {GGML_MODEL_PATH}, skipping download)")
        return
    py, env = _hf_download_env(ctx)
    subprocess.run([
        py, "-u", "-c",
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id={GGML_MODEL_HF_REPO!r}, "
        f"filename={GGML_MODEL_FILENAME!r}, "
        f"local_dir={str(GGML_MODEL_PATH.parent)!r})",
    ], check=True, env=env)


def next_preload_engines(current: str, engine_id: str) -> str | None:
    normalized = current.strip()
    token = f"'{engine_id}'"
    if token in normalized:
        return None
    if normalized in {"", "[]", "@as []"}:
        return f"[{token}]"
    clean = normalized.removeprefix("@as ").strip()
    if not clean.endswith("]"):
        raise ValueError(f"Unexpected preload-engines value: {current!r}")
    return f"{clean[:-1]}, {token}]"


def previous_preload_engines(current: str, engine_id: str) -> str | None:
    """Return the preload list with *engine_id* removed, or None if absent.

    Used by ``reset-kdictate-install.sh`` to cleanly remove KDictate
    without wiping other IBus engines the user had configured.
    """
    normalized = current.strip()
    token = f"'{engine_id}'"
    if not normalized or normalized in {"[]", "@as []"} or token not in normalized:
        return None
    clean = normalized.removeprefix("@as ").strip()
    if not (clean.startswith("[") and clean.endswith("]")):
        raise ValueError(f"Unexpected preload-engines value: {current!r}")
    parts = [p.strip() for p in clean[1:-1].split(",")]
    remaining = [p for p in parts if p and p != token]
    return f"[{', '.join(remaining)}]" if remaining else "@as []"


def configure_preload_engines(ctx: InstallContext) -> None:
    result = run_command(
        ["dconf", "read", "/desktop/ibus/general/preload-engines"],
        quiet=True, check=False,
    )
    if result.returncode != 0:
        return
    try:
        new = next_preload_engines(result.stdout.strip(), DBUS_INTERFACE)
    except ValueError as exc:
        log(f"skipping preload-engines: {exc}")
        return
    if new is not None:
        run_command(["dconf", "write", "/desktop/ibus/general/preload-engines", new])


def configure_kwin_input_method(ctx: InstallContext) -> None:
    if shutil.which("kwriteconfig6") is None:
        return
    if KDE_VIRTUAL_KEYBOARD_DESKTOP.is_file():
        run_command([
            "kwriteconfig6", "--file", ctx.home / ".config" / "kwinrc",
            "--group", "Wayland", "--key", "InputMethod",
            KDE_VIRTUAL_KEYBOARD_DESKTOP,
        ])
    else:
        log(f"Warning: {KDE_VIRTUAL_KEYBOARD_DESKTOP} not found")
    run_command([
        "kwriteconfig6", "--file", ctx.home / ".config" / "kwinrc",
        "--group", "Wayland", "--key", "VirtualKeyboardEnabled", "true",
    ])


def register_global_shortcut(ctx: InstallContext) -> None:
    """Ensure the Ctrl+Space binding exists *and* is correct.

    Treating the section's presence as sufficient made this unrepairable in
    the case that most needs repairing. Plasma rewrites kglobalshortcutsrc on
    its own — a shortcut reset or a conflict can drop or change the ``_launch``
    line while leaving the section behind — and the binding is then broken
    with no way for ``--reconfigure`` to restore it, which is precisely what
    that flag advertises.
    """

    shortcut_file = ctx.home / ".config" / "kglobalshortcutsrc"
    section = f"[services][{TOGGLE_DESKTOP_NAME}]"
    entry = "_launch=Ctrl+Space, Ctrl+Space"
    content = shortcut_file.read_text(encoding="utf-8") if shortcut_file.exists() else ""

    if section not in content:
        content = content.rstrip("\n") + f"\n\n{section}\n{entry}\n"
        write_home_file(ctx, shortcut_file, content)
        return

    out: list[str] = []
    in_section = False
    has_entry = False
    changed = False

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if in_section and not has_entry:
                # Section ended without a _launch line: restore it.
                out.append(entry)
                changed = True
                has_entry = True
            in_section = stripped == section
        elif in_section and stripped.startswith("_launch="):
            has_entry = True
            if stripped != entry:
                line = entry
                changed = True
        out.append(line)

    if in_section and not has_entry:
        out.append(entry)
        changed = True

    if not changed:
        return
    write_home_file(ctx, shortcut_file, "\n".join(out) + "\n")


_ENGINE_PROCESS_PATTERN = "(libexec|bin)/ibus-engine-kdictate"


def clear_stale_engine_processes() -> int:
    """Terminate kdictate IBus engine processes so IBus respawns them fresh.

    An engine process reads its script once at spawn time and keeps executing
    what it loaded. Upgrading the package replaces that file on disk without
    touching the running process, so afterwards the live engine is still
    running the *previous* version's code — and the copy IBus spawns for the
    new session sits alongside it. Both are on the session bus receiving the
    daemon's ``FinalTranscript`` broadcast, while only one can hold the
    focused input context, so a transcript can be delivered to an engine that
    is not the one able to commit it.

    The ``pkill -x ibus-daemon`` further down does not reach these: ``-x``
    matches the process name exactly, and an engine runs as
    ``python …/ibus-engine-kdictate``. Killing the daemon *orphans* the
    engines rather than reaping them — which is why they also survive an
    ``ibus restart``, observed here with an engine that outlived both.

    Returns how many processes were signalled. Safe to call whenever the
    on-disk engine has changed: IBus spawns an engine on demand, so the next
    activation gets a fresh one running the installed code.
    """

    if shutil.which("pgrep") is None or shutil.which("pkill") is None:
        return 0

    uid = str(os.getuid())
    # Restricted to this user, and matched on the executed path rather than a
    # bare name, so nothing outside this install's own engines is touched.
    select = ["-u", uid, "-f", _ENGINE_PROCESS_PATTERN]

    found = run_command(["pgrep", *select], quiet=True, check=False)
    pids = [tok for tok in found.stdout.split() if tok.isdigit()]
    if not pids:
        install_log("no stale engine processes")
        return 0

    install_log(f"killing stale engine process(es) pid {','.join(pids)}")
    run_command(["pkill", *select], quiet=True, check=False)
    time.sleep(0.5)

    survivors = run_command(["pgrep", *select], quiet=True, check=False)
    survivor_pids = [tok for tok in survivors.stdout.split() if tok.isdigit()]
    if survivor_pids:
        install_log(f"SIGKILL for surviving engine process(es) pid {','.join(survivor_pids)}")
        run_command(["pkill", "-9", *select], quiet=True, check=False)

    return len(pids)


# The Wayland IM bridge KWin spawns from the kwinrc [Wayland] InputMethod
# desktop file. Its --exec-daemon child *is* the session's ibus-daemon; the
# two live and die together.
_WAYLAND_BRIDGE_PATTERN = "ibus-ui-gtk3.*--enable-wayland-im"


def _pgrep_pids(pattern: str, *, exact_name: bool = False) -> list[str]:
    """Return this user's PIDs whose command line (or name) matches."""

    if shutil.which("pgrep") is None:
        return []
    match_flag = "-x" if exact_name else "-f"
    result = run_command(
        ["pgrep", "-u", str(os.getuid()), match_flag, pattern],
        quiet=True, check=False,
    )
    return [tok for tok in result.stdout.split() if tok.isdigit()]


def _wayland_bridge_pids() -> list[str]:
    return _pgrep_pids(_WAYLAND_BRIDGE_PATTERN)


def _ibus_daemon_pids() -> list[str]:
    return _pgrep_pids("ibus-daemon", exact_name=True)


def _kwin_wayland_running() -> bool:
    return bool(_pgrep_pids("kwin_wayland", exact_name=True))


def _qdbus_bin() -> str | None:
    for candidate in ("qdbus6", "qdbus"):
        if shutil.which(candidate) is not None:
            return candidate
    return None


def refresh_ibus_registry(ctx: InstallContext) -> None:
    """Refresh IBus after the on-disk engine changed — without breaking it.

    History, because this function has now caused two distinct outages:

    Earlier versions killed ibus-daemon here and toggled KWin's
    ``org.kde.kwin.VirtualKeyboard.enabled`` property to make KWin respawn
    the whole input-method stack. Plasma 6.7 removed that property, so the
    toggle failed on every install *after* the daemon was already dead, and
    the fallback started a bare ``ibus-daemon -r -d`` with no Wayland IM
    bridge. Result (diagnosed 2026-08-10): the engine received transcripts
    and committed them, but ``ss -xp`` showed no compositor or application
    attached to the new daemon's socket — every commit vanished, and only a
    relogin (which lets KWin respawn the bridge) recovered typing. This is
    why "install → broken, reboot → fixed" recurred across 0.14–0.16.

    The rules now:

    * NEVER kill a healthy ibus-daemon. IBus spawns engine processes on
      demand, so shipping new engine code needs only a registry write-cache
      plus killing the old engine processes.
    * Only when the Wayland IM bridge is already missing — the session's
      typing path is already dead, nothing left to protect — attempt a
      KWin-driven relaunch of the full stack (see repair_wayland_im_bridge).
    """

    ibus_env = {
        "IBUS_COMPONENT_PATH": (
            f"{ctx.home / '.local/share/ibus/component'}:/usr/share/ibus/component"
        )
    }
    # Update IBus component registry so ibus-daemon knows about the new engine.
    run_command(["ibus", "write-cache"], env=ibus_env, quiet=True)
    install_log("ibus write-cache completed")

    # Clear engine processes left over from before this upgrade so the next
    # activation spawns a fresh engine running the just-installed code.
    clear_stale_engine_processes()

    bridge = _wayland_bridge_pids()
    daemons = _ibus_daemon_pids()

    if not _kwin_wayland_running():
        # Non-KDE-Wayland session: there is no bridge to manage. Make sure a
        # daemon exists for the cold-start case and leave everything else
        # alone.
        if not daemons:
            install_log("no ibus-daemon and no kwin_wayland; starting plain ibus-daemon")
            run_command(["ibus-daemon", "-r", "-d"], quiet=True, check=False)
        return

    if bridge and daemons:
        install_log(
            f"IM stack healthy (bridge pid {','.join(bridge)}, "
            f"ibus-daemon pid {','.join(daemons)}); leaving it untouched"
        )
        return

    install_log(
        f"IM stack incomplete (bridge={bridge or 'none'}, "
        f"ibus-daemon={daemons or 'none'}); attempting KWin relaunch"
    )
    repair_wayland_im_bridge(ctx)


def repair_wayland_im_bridge(ctx: InstallContext) -> None:
    """Make KWin relaunch its input-method stack (bridge + ibus-daemon).

    KWin (re)launches the command named by kwinrc ``[Wayland] InputMethod``
    on reconfigure whenever the value *changes*. Flipping the key off and
    back on is therefore the supported no-logout path to a fresh
    ``ibus-ui-gtk3 --enable-wayland-im --exec-daemon`` stack — the same
    mechanism the Virtual Keyboard KCM uses when the user switches input
    methods. (The old ``VirtualKeyboard.enabled`` D-Bus toggle is gone as
    of Plasma 6.7; do not resurrect it.)

    Only called when the bridge is already missing, i.e. the session's
    typing path is already severed — so the daemon kill below cannot make
    anything worse than it already is.
    """

    qdbus_bin = _qdbus_bin()
    if qdbus_bin is None or shutil.which("kwriteconfig6") is None:
        install_log("bridge repair skipped: qdbus6/kwriteconfig6 unavailable")
        return
    if not KDE_VIRTUAL_KEYBOARD_DESKTOP.is_file():
        install_log(f"bridge repair skipped: {KDE_VIRTUAL_KEYBOARD_DESKTOP} missing")
        return

    # KWin must be reachable on the session bus for any of this to work.
    probe = run_command(
        [qdbus_bin, "org.kde.KWin", "/KWin", "reconfigure"],
        quiet=True, check=False,
    )
    if probe.returncode != 0:
        install_log("bridge repair skipped: KWin not reachable on the session bus")
        return

    # A bare ibus-daemon (the broken-session shape this repairs) would
    # collide with the daemon the relaunched bridge execs. Clear it first.
    stale = _ibus_daemon_pids()
    if stale:
        if shutil.which("pkill") is None:
            install_log("bridge repair skipped: stale ibus-daemon present but pkill unavailable")
            return
        install_log(f"killing bridgeless ibus-daemon pid {','.join(stale)}")
        run_command(["pkill", "-u", str(os.getuid()), "-x", "ibus-daemon"],
                    quiet=True, check=False)
        time.sleep(0.5)

    kwinrc = ctx.home / ".config" / "kwinrc"
    run_command([
        "kwriteconfig6", "--file", kwinrc,
        "--group", "Wayland", "--key", "InputMethod", "--delete",
    ], quiet=True, check=False)
    run_command([qdbus_bin, "org.kde.KWin", "/KWin", "reconfigure"],
                quiet=True, check=False)
    time.sleep(1.0)
    run_command([
        "kwriteconfig6", "--file", kwinrc,
        "--group", "Wayland", "--key", "InputMethod",
        KDE_VIRTUAL_KEYBOARD_DESKTOP,
    ], quiet=True, check=False)
    run_command([qdbus_bin, "org.kde.KWin", "/KWin", "reconfigure"],
                quiet=True, check=False)

    # KWin spawns the bridge asynchronously; give it a bounded window.
    for _ in range(20):
        bridge = _wayland_bridge_pids()
        if bridge:
            install_log(f"bridge repair succeeded (bridge pid {','.join(bridge)})")
            return
        time.sleep(0.5)

    install_log("bridge repair FAILED: no bridge process appeared within 10s")
    # Last resort so basic IBus (X11/XWayland clients) still works; the
    # Wayland typing path stays down until the next login.
    if not _ibus_daemon_pids():
        run_command(["ibus-daemon", "-r", "-d"], quiet=True, check=False)
        install_log("started plain ibus-daemon as fallback (no Wayland bridge)")


def check_im_stack() -> list[str]:
    """Verify the full input-method chain after an install.

    Every prior outage in this project shipped behind a green checklist:
    the installer verified its own steps but never the state it left the
    session in. This check covers the chain a dictated transcript must
    traverse — daemon bus name → engine process → active engine →
    ibus-daemon → Wayland bridge — and returns human-readable problems
    instead of letting the summary claim success over a severed stack.
    """

    problems: list[str] = []

    if not _ibus_daemon_pids():
        problems.append("ibus-daemon is not running")

    if _kwin_wayland_running() and not _wayland_bridge_pids():
        problems.append(
            "the Wayland IM bridge (ibus-ui-gtk3 --enable-wayland-im) is not "
            "running — committed text cannot reach applications until you "
            "log out and back in"
        )

    active = run_command(["ibus", "engine"], quiet=True, check=False)
    active_name = active.stdout.strip()
    if active.returncode != 0 or active_name != DBUS_INTERFACE:
        problems.append(
            f"active IBus engine is {active_name or 'unknown'!r}, "
            f"expected {DBUS_INTERFACE!r}"
        )

    if not _pgrep_pids(_ENGINE_PROCESS_PATTERN):
        problems.append("no kdictate engine process is running")

    if shutil.which("gdbus") is not None:
        owner = run_command(
            ["gdbus", "call", "--session",
             "--dest", "org.freedesktop.DBus",
             "--object-path", "/org/freedesktop/DBus",
             "--method", "org.freedesktop.DBus.NameHasOwner", DBUS_BUS_NAME],
            quiet=True, check=False,
        )
        if owner.returncode == 0 and "true" not in owner.stdout:
            problems.append(
                f"kdictate daemon does not own {DBUS_BUS_NAME} on the session bus"
            )

    return problems


def reload_systemd_user(ctx: InstallContext) -> None:
    run_command(["systemctl", "--user", "daemon-reload"], quiet=True)
    run_command(["systemctl", "--user", "enable", SERVICE_NAME], quiet=True)
    run_command(["systemctl", "--user", "restart", SERVICE_NAME], quiet=True)


def _is_packaged_install() -> bool:
    """True when kdictate is installed as a system package.

    The package provides the runtime, the /usr/bin launchers, and the
    system-level integration files (systemd unit, D-Bus service, IBus
    component, env.d). When present, the installer runs in *configurator*
    mode: it downloads the model, wires the per-user KDE bits, and enables
    the package's system service -- it does NOT build a venv or install
    per-user copies that would shadow the package.
    """
    return (
        Path("/usr/lib/kdictate/bin/whisper-cli").exists()
        and (Path("/usr/lib/systemd/user") / SERVICE_NAME).exists()
    )


def _installed_version(ctx: InstallContext, packaged: bool) -> str | None:
    """Return the version currently installed, or None if there is none.

    A packaged install is read from the package manager rather than by
    importing the installed module: the 0.12.0 package shipped a wheel whose
    ``APP_VERSION`` still read 0.11.1, so the installed code's own string is
    not a reliable record of what was actually put on disk. ``pacman -Q`` is.
    """

    if packaged:
        result = run_command(
            ["pacman", "-Q", PACKAGE_NAME], quiet=True, check=False,
        )
        if result.returncode != 0:
            return None
        fields = result.stdout.split()
        if len(fields) < 2:
            return None
        # "kdictate 0.13.0-1" -> "0.13.0"; the pkgrel is not part of our
        # version lockstep and would never compare equal to __version__.
        return fields[1].rsplit("-", 1)[0]

    if not ctx.python_bin.exists():
        return None
    # -I (isolated mode) so the probe answers about the *installed* copy and
    # nothing else. `python -c` otherwise resolves sys.path[0] to the cwd —
    # which for the documented `python3 install.py` from the repo root is the
    # source tree itself — and additionally honours PYTHONPATH, which a
    # development shell or IDE may well have pointing at this same checkout.
    # Either route makes the probe import the tree it is installing *from*,
    # report that version as installed, and turn the version gate below into
    # a permanent "already up to date" no-op. -I covers both (it implies -P,
    # -E and -s); cwd is moved out of the repo as belt and braces so the
    # answer cannot depend on where the installer was invoked.
    result = run_command(
        [ctx.python_bin, "-I", "-c",
         "from kdictate import __version__; print(__version__)"],
        quiet=True, check=False, cwd=ctx.home,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _prompt_update(installed: str | None) -> bool:
    """Show the version transition and confirm it. True means proceed."""

    if installed is None:
        print(f"  No existing install detected — installing {__version__}.\n")
        return True

    print(f"    Installed:  {installed}")
    print(f"    This tree:  {__version__}\n")
    while True:
        choice = input(
            f"  Update {installed} → {__version__}? [Y/n]: "
        ).strip().lower()
        if choice in ("", "y", "yes"):
            return True
        if choice in ("n", "no"):
            return False


def _die_with_output(
    message: str, result: subprocess.CompletedProcess[str], tail: int = 25,
) -> NoReturn:
    """Fail with the tail of a suppressed command's output.

    Every step here runs quiet, which is only safe if a failure still hands
    back something actionable. Printing the tail rather than the whole log
    keeps the successful path to one screen without turning a real build
    error into "it failed, go find out why yourself".
    """

    combined = (result.stdout or "") + (result.stderr or "")
    lines = [line for line in combined.splitlines() if line.strip()]
    shown = "\n".join(f"      {line}" for line in lines[-tail:]) or "      (no output)"
    die(f"{message} (exit {result.returncode}). Last {min(tail, len(lines))} lines:\n\n{shown}")


def _require_build_dependencies() -> list[str]:
    """Return the package names makepkg needs that are not importable.

    Pure detection — see :func:`ensure_build_dependencies` for the part that
    acts on the answer.
    """

    # -I (isolated mode) and a cwd outside the checkout, for the same reason
    # _installed_version needs them. This runs from a repo root that
    # accumulates a `build/` directory from the very wheel step these
    # dependencies exist to perform, and `python -c` both puts the cwd on
    # sys.path and honours PYTHONPATH. Either route lets Python import that
    # directory as an implicit namespace package, so `import build` succeeds
    # against a pile of build artifacts and the check reports python-build
    # present when pacman has never heard of it -- the user installs only what
    # was reported, re-runs, and makepkg dies at `python -m build` anyway.
    # -I closes both (it implies -P, -E and -s).
    missing = [
        package
        for module, package in (("build", "python-build"), ("installer", "python-installer"))
        if run_command(
            [sys.executable, "-I", "-c", f"import {module}"],
            quiet=True, check=False, cwd=Path.home(),
        ).returncode != 0
    ]
    return missing


def ensure_build_dependencies() -> None:
    """Install the rebuild's build dependencies, with consent.

    Called during preflight rather than from inside the rebuild step, so the
    pacman transaction and its password prompt happen in the open, before the
    quiet one-screen checklist starts — not buried inside a step whose output
    is suppressed. That was why ``makepkg --syncdeps`` was dropped; refusing
    to install them at all was an overcorrection that left the user to run a
    command and start over.

    Installed with ``--asdeps``, because that is what they are: tooling this
    build needs, not something the user asked to have on their system.
    Marking them explicit would keep an orphan sweep from ever reclaiming
    them, which is the installer overriding the machine's cleanup policy to
    protect itself from a round trip. Whether build tooling stays installed
    is the user's call — ``makepkg --rmdeps`` and ``yay --removemake`` exist
    precisely so it can be — and this only has to guarantee that a sweep
    never leaves the next rebuild dead-ended, which re-installing on demand
    already does.
    """

    missing = _require_build_dependencies()
    if not missing:
        return

    distro = _detect_distro()
    hint = _pkg_hint(distro, " ".join(missing))
    print("  The package rebuild needs these build dependencies:\n")
    for package in missing:
        print(f"      {package}")
    print()

    if distro != "arch" or shutil.which("sudo") is None:
        die(f"Install them first:\n\n      {hint}\n\n      Then re-run this installer.")

    while True:
        choice = input("  Install them now? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            break
        if choice in ("n", "no"):
            die(f"Install them first:\n\n      {hint}\n\n      Then re-run this installer.")

    print()
    result = run_command(
        ["sudo", "pacman", "-S", "--needed", "--asdeps", *missing], check=False,
    )
    print()
    if result.returncode != 0:
        die(f"Installing the build dependencies failed.\n\n      Try:  {hint}")

    still_missing = _require_build_dependencies()
    if still_missing:
        die(
            "These are still not importable after installing:\n\n"
            + "".join(f"        {package}\n" for package in still_missing)
        )


def rebuild_and_install_package(ctx: InstallContext) -> None:
    """Rebuild the package from this tree and install it.

    Configurator mode deliberately never writes Python code — the package
    owns ``/usr/lib/python3.*/site-packages/kdictate``. So on a packaged
    system a rebuild is the *only* way this tree's code reaches the running
    daemon; without it the installer would wire up the KDE bits, restart the
    service on whatever the package already contained, and report success
    while the fix the user came here for was never deployed.

    Both commands run quiet, like every other step, with their output
    surfaced only if they fail. pacman is given the built package explicitly
    rather than by glob-of-latest, so a stale build from an earlier version
    can never be installed by accident.
    """

    for tool in ("makepkg", "pacman", "sudo"):
        require_command(tool)
    # Belt and braces: ensure_build_dependencies() already resolved these
    # during preflight, where a pacman prompt can be shown in the open. If
    # anything is still missing by the time we get here, fail now rather than
    # several minutes into a build that cannot succeed.
    missing = _require_build_dependencies()
    if missing:
        die(
            "Missing build dependencies for the package rebuild:\n\n"
            + "".join(f"        {package}\n" for package in missing)
            + f"\n      Install:  {_pkg_hint(_detect_distro(), ' '.join(missing))}"
        )

    pkg_dir = ctx.script_dir / "packaging"
    result = run_command(["makepkg", "--force"], cwd=pkg_dir, quiet=True, check=False)
    if result.returncode != 0:
        _die_with_output("Package build failed", result)

    built = sorted(
        pkg_dir.glob(f"{PACKAGE_NAME}-{__version__}-*.pkg.tar.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not built:
        die(
            f"makepkg finished but produced no {PACKAGE_NAME} {__version__} "
            f"package in {pkg_dir}.\n\n"
            "      Check that packaging/PKGBUILD's pkgver matches "
            f"kdictate.__version__ ({__version__})."
        )

    # Ask for credentials up front and visibly. Left to the pacman call
    # below, the password prompt would appear inside a suppressed step with
    # no indication of what was waiting on it.
    if run_command(["sudo", "-v"], check=False).returncode != 0:
        die("sudo authentication failed; the new package was built but not installed.")

    # Already confirmed at the update prompt; --noconfirm avoids asking twice.
    result = run_command(
        ["sudo", "pacman", "-U", "--noconfirm", built[-1]], quiet=True, check=False,
    )
    if result.returncode != 0:
        _die_with_output(f"Installing {built[-1].name} failed", result)


def _cleanup_shadowing_user_setup(ctx: InstallContext) -> None:
    """Remove per-user files that would shadow the system package.

    A prior source install (or earlier run) leaves per-user systemd/D-Bus/
    IBus units pointing at a venv; left in place they override the package's
    system units, so the venv daemon runs instead of the packaged one.
    Remove them and the now-redundant venv so the package is authoritative.
    The model and other runtime state are left intact.
    """
    for path in (
        ctx.home / ".config/systemd/user" / SERVICE_NAME,
        ctx.home / ".local/share/dbus-1/services" / DBUS_SERVICE_NAME,
        ctx.home / ".local/share/ibus/component" / IBUS_COMPONENT_NAME,
        ctx.home / ".config/environment.d" / IBUS_ENV_FILE_NAME,
    ):
        if path.is_symlink() or path.exists():
            path.unlink()
            log(f"removed shadowing per-user file: {path}")
    if ctx.venv_dir.exists():
        shutil.rmtree(ctx.venv_dir, ignore_errors=True)
        log(f"removed redundant venv: {ctx.venv_dir}")


def _write_backend_dropin(ctx: InstallContext) -> None:
    """Pin the package's system service to the install-time backend choice.

    The package unit ships a default backend flag; this systemd drop-in
    overrides ExecStart with the explicit gpu/cpu chosen here, so a packaged
    install runs exactly one backend -- never auto/both, no runtime fallback.
    """
    if ctx.gpu:
        exec_args = "--backend gpu"
    else:
        # The packaged daemon's default CT2 model_dir is PROJECT_ROOT-relative
        # (site-packages), not the runtime dir where install.py downloads it,
        # so packaged CPU mode must be pointed at the model explicitly.
        exec_args = f"--backend cpu --model-dir {ctx.runtime_dir / DEFAULT_MODEL_NAME}"
    dropin = ctx.home / ".config/systemd/user" / f"{SERVICE_NAME}.d" / "10-backend.conf"
    write_home_file(
        ctx, dropin,
        "[Service]\n"
        "ExecStart=\n"
        f"ExecStart=/usr/bin/kdictate-daemon {exec_args}\n",
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the installer's own options."""

    parser = argparse.ArgumentParser(
        prog="install.py",
        description="Install or update KDictate for the current user.",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Re-run the configuration steps even when the installed version "
             "already matches this tree. Use this to repair a broken install "
             "-- a clobbered Ctrl+Space binding, a missing IBus engine "
             "registration, a backend switch -- without having to change the "
             "version number first.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if os.geteuid() == 0:
        die(
            "Run as your user, not root.\n\n"
            "      KDictate installs under ~/. No root needed:\n\n"
            "        python3 install.py"
        )

    script_path = Path(__file__).resolve()
    ctx = InstallContext(
        script_path=script_path,
        script_dir=script_path.parent,
        home=Path.home(),
        runtime_dir=Path.home() / ".local" / "share" / "kdictate",
    )

    print(f"\n  KDictate {__version__} installer\n")

    # Version gate first, before any other prompt: a system that is already
    # current should cost one command and no questions.
    #
    # --reconfigure exists because every configuration step below is
    # idempotent and re-running them is the documented repair path (a Plasma
    # reset clobbers the Ctrl+Space binding, the IBus preload list loses its
    # entry, the backend needs switching). Gating those behind a version
    # change would leave a user whose install is broken *at the current
    # version* with no way to fix it short of editing app_metadata.py, so the
    # exit below names the flag rather than just refusing.
    packaged = _is_packaged_install()
    installed = _installed_version(ctx, packaged)
    up_to_date = installed is not None and installed == __version__

    if up_to_date and not args.reconfigure:
        print(f"  Already at {__version__} — nothing to do.")
        print("  Re-run with --reconfigure to repair the KDE/IBus wiring.\n")
        return 0

    if up_to_date:
        print(f"  Already at {__version__}; reconfiguring at your request.\n")
    elif not _prompt_update(installed):
        print("\n  Cancelled — nothing was changed.\n")
        return 0

    install_log(
        f"install started: {installed or 'none'} -> {__version__}"
        f" ({'packaged' if packaged else 'source'}"
        f"{', reconfigure' if up_to_date else ''})"
    )

    # Reconfiguring at the same version has nothing to rebuild: the installed
    # package already contains this tree's code, and a rebuild is the single
    # most expensive step there is (whisper.cpp + Vulkan, minutes).
    rebuild = packaged and not up_to_date

    # The two modes run different step sets; the count was previously
    # hardcoded to the source-install total, so a packaged run counted up to
    # 8/11 and stopped.
    global _TOTAL_STEPS  # noqa: PLW0603
    _TOTAL_STEPS = (10 if rebuild else 9) if packaged else 12

    gpu = _prompt_backend()
    if gpu:
        ctx = InstallContext(
            script_path=ctx.script_path, script_dir=ctx.script_dir,
            home=ctx.home, runtime_dir=ctx.runtime_dir, gpu=True,
        )

    print()
    if packaged and rebuild:
        log("Packaged install detected — rebuilding the system package from "
            "this tree, then configuring it (model + per-user KDE wiring + "
            "system service; no venv).")
    elif packaged:
        log("Packaged install detected — reconfiguring only (model + per-user "
            "KDE wiring + system service); the package is already current.")

    preflight_ibus()
    required = ["python3", "systemctl", "dconf"]
    if not packaged:
        required.insert(2, "rsync")
    for cmd in required:
        require_command(cmd)

    # Before the checklist, so the pacman transaction and its password prompt
    # happen in the open rather than inside a step whose output is suppressed.
    if rebuild:
        ensure_build_dependencies()

    pkg = ctx.script_dir / "packaging"

    if rebuild:
        # The step that actually moves code onto the system, and the long one
        # (whisper.cpp + Vulkan). Signposted in the label because it runs
        # quiet like the rest, so there is nothing else to watch.
        step("Rebuilding system package (several minutes)")
        rebuild_and_install_package(ctx)
        step_done(f"{PACKAGE_NAME} {__version__}")

    if not packaged:
        step("Syncing runtime files")
        sync_runtime(ctx)
        step_done()

        step("Setting up Python environment")
        install_python_environment(ctx)
        step_done()

    if gpu:
        step("Downloading GPU model")
        download_gpu_model(ctx)
        step_done(GGML_MODEL_HF_REPO)
    else:
        step("Downloading CPU model")
        download_cpu_model(ctx)
        step_done(DEFAULT_MODEL_HF_REPO)

    if packaged:
        step("Clearing stale per-user setup")
        _cleanup_shadowing_user_setup(ctx)
        step_done()

        step("Pinning backend")
        _write_backend_dropin(ctx)
        step_done(f"--backend {'gpu' if gpu else 'cpu'}")
    else:
        # A prior packaged/configurator run leaves a backend drop-in that would
        # override this venv-backed unit; clear it before writing the source one.
        dropin_dir = ctx.home / ".config/systemd/user" / f"{SERVICE_NAME}.d"
        if dropin_dir.exists():
            shutil.rmtree(dropin_dir, ignore_errors=True)

        step("Installing systemd user service")
        install_rendered_file(ctx, pkg / "kdictate-systemd.service",
                              ctx.home / ".config/systemd/user" / SERVICE_NAME)
        step_done()

        step("Installing D-Bus activation service")
        install_rendered_file(ctx, pkg / f"{APP_ROOT_ID}.service",
                              ctx.home / ".local/share/dbus-1/services" / DBUS_SERVICE_NAME)
        step_done()

        step("Installing IBus engine metadata")
        install_rendered_file(ctx, pkg / IBUS_COMPONENT_NAME,
                              ctx.home / ".local/share/ibus/component" / IBUS_COMPONENT_NAME)
        install_rendered_file(ctx, pkg / IBUS_ENV_FILE_NAME,
                              ctx.home / ".config/environment.d" / IBUS_ENV_FILE_NAME)
        step_done()

    step("Installing KDE/Plasma integration")
    copy_home_file(ctx, pkg / PLASMA_ENV_SCRIPT_NAME,
                   ctx.home / ".config/plasma-workspace/env" / PLASMA_ENV_SCRIPT_NAME)
    if not packaged:
        install_rendered_file(ctx, pkg / TOGGLE_DESKTOP_NAME,
                              ctx.home / ".local/share/applications" / TOGGLE_DESKTOP_NAME)
    if shutil.which("kbuildsycoca6") is not None:
        run_command(["kbuildsycoca6", "--noincremental"], quiet=True, check=False)
    register_global_shortcut(ctx)
    step_done()

    step("Registering IBus input method")
    configure_preload_engines(ctx)
    configure_kwin_input_method(ctx)
    step_done()

    step("Refreshing IBus engine registry")
    refresh_ibus_registry(ctx)
    step_done()

    step("Starting KDictate service")
    reload_systemd_user(ctx)
    step_done()

    step("Activating KDictate input method")
    for _ in range(5):
        if run_command(["ibus", "engine", DBUS_INTERFACE], quiet=True, check=False).returncode == 0:
            break
        time.sleep(1)
    step_done()

    step("Verifying input-method stack")
    problems = check_im_stack()
    step_done("healthy" if not problems else f"{len(problems)} issue(s)")
    install_log(
        "self-check: " + ("healthy" if not problems else "; ".join(problems))
    )

    verb = "configured" if packaged else "installed"
    mode = "GPU (Vulkan)" if gpu else "CPU (faster-whisper)"
    if problems:
        print(f"\n  ⚠️  KDictate {__version__} {verb} ({mode}), but the "
              "input-method stack has problems:\n")
        for problem in problems:
            print(f"     - {problem}")
        print(f"\n     Details: {INSTALL_LOG_PATH}\n")
    else:
        print(f"\n  \U0001f389 KDictate {__version__} {verb} ({mode})")
        print("     Ctrl+Space to toggle dictation.\n")
    return 0


def preflight_ibus() -> None:
    missing = [cmd for cmd in ("ibus", "ibus-daemon") if shutil.which(cmd) is None]
    if not missing:
        return
    distro = _detect_distro()
    die(
        "KDictate needs ibus and ibus-daemon.\n\n"
        f"      Install:  {_pkg_hint(distro, 'ibus')}\n\n"
        "      (Missing: " + ", ".join(missing) + ")"
    )


if __name__ == "__main__":
    raise SystemExit(main())
