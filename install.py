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
from kdictate.constants import APP_ROOT_ID, DBUS_INTERFACE

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
            "@@BACKEND_FLAGS@@": " --backend gpu" if self.gpu else "",
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

    # A packaged install ships a vendored whisper-cli under the private
    # prefix; prefer it, then fall back to PATH for source/dev checkouts.
    vendored = Path("/usr/lib/kdictate/bin/whisper-cli")
    binary = (
        str(vendored) if (vendored.is_file() and os.access(vendored, os.X_OK))
        else (shutil.which("whisper-cli") or shutil.which("whisper-cpp"))
    )
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
) -> subprocess.CompletedProcess[str]:
    args = [str(p) for p in command]
    proc_env = {**os.environ, **(env or {})}
    return subprocess.run(
        args, check=check, encoding="utf-8", errors="replace",
        capture_output=quiet, env=proc_env, timeout=timeout,
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
    subprocess.run([
        str(ctx.python_bin), "-u", "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={DEFAULT_MODEL_HF_REPO!r}, "
        f"local_dir={str(model_dir)!r}, "
        f"max_workers=1)",
    ], check=True)


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
    subprocess.run([
        str(ctx.python_bin), "-u", "-c",
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id={GGML_MODEL_HF_REPO!r}, "
        f"filename={GGML_MODEL_FILENAME!r}, "
        f"local_dir={str(GGML_MODEL_PATH.parent)!r})",
    ], check=True)


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
    shortcut_file = ctx.home / ".config" / "kglobalshortcutsrc"
    section = f"[services][{TOGGLE_DESKTOP_NAME}]"
    entry = "_launch=Ctrl+Space, Ctrl+Space"
    content = shortcut_file.read_text(encoding="utf-8") if shortcut_file.exists() else ""
    if section in content:
        return
    content = content.rstrip("\n") + f"\n\n{section}\n{entry}\n"
    write_home_file(ctx, shortcut_file, content)


def refresh_ibus_registry(ctx: InstallContext) -> None:
    ibus_env = {
        "IBUS_COMPONENT_PATH": (
            f"{ctx.home / '.local/share/ibus/component'}:/usr/share/ibus/component"
        )
    }
    # Update IBus component registry so ibus-daemon knows about the new engine.
    run_command(["ibus", "write-cache"], env=ibus_env, quiet=True)

    # Tell KWin to re-read kwinrc so it picks up the InputMethod key written
    # by configure_kwin_input_method in the same install run.  Without this,
    # KWin still has a null InputMethod in memory and the toggle below does
    # nothing because there's no input method desktop file to launch.
    qdbus_bin: str | None = None
    for candidate in ("qdbus6", "qdbus"):
        if shutil.which(candidate) is not None:
            qdbus_bin = candidate
            break
    if qdbus_bin is None:
        # Without qdbus6/qdbus we have no recovery path: killing the daemon
        # below would leave the user with a broken IM stack and no way to
        # bring it back without logout.  Skip the hot-start entirely; the
        # next session login picks up the new InputMethod desktop file.
        return

    # KWin must be reachable on the session bus for the toggle below to do
    # anything useful.  If reconfigure fails (non-KDE session, transient D-Bus
    # failure, etc.), there's no point killing the daemon — the toggle won't
    # recover it either, and we'd leave the user worse off than we found them.
    reconfigure = run_command(
        [qdbus_bin, "org.kde.KWin", "/KWin", "reconfigure"],
        quiet=True, check=False,
    )
    if reconfigure.returncode != 0:
        return

    # Kill any stale ibus-daemon so the toggle below has a clean slate.  KWin
    # only spawns ibus-ui-gtk3 --enable-wayland-im on a true cold-start of the
    # input method; if a daemon is already registered on the session bus
    # (especially one started with --panel disable), the toggle no-ops and we
    # end up with a daemon but no Wayland IM bridge.  If pkill is unavailable
    # (unusual but possible on minimal containers), there's no way to clear a
    # stale daemon and the toggle would no-op — skip the hot-start so the
    # next session login picks up the new InputMethod cleanly.
    if shutil.which("pkill") is None:
        return
    run_command(["pkill", "-x", "ibus-daemon"], quiet=True, check=False)
    time.sleep(0.5)

    # Toggle KWin's VirtualKeyboard.enabled from false to true.  This is the
    # signal that makes KWin invoke the InputMethod desktop file:
    #   ibus-ui-gtk3 --enable-wayland-im --exec-daemon …
    # which spawns both the daemon and the Wayland IM bridge in one shot.
    # Empirically, "ibus restart" does NOT trigger this — it re-execs the
    # daemon in place with the same args, so KWin sees no D-Bus name change
    # and the bridge is never launched.
    toggle_ok = True
    for value in ("false", "true"):
        result = run_command(
            [qdbus_bin, "--literal", "org.kde.KWin", "/VirtualKeyboard",
             "org.freedesktop.DBus.Properties.Set",
             "org.kde.kwin.VirtualKeyboard", "enabled", value],
            quiet=True, check=False,
        )
        if result.returncode != 0:
            toggle_ok = False
            break
        time.sleep(0.5)

    # If the toggle failed after we killed the daemon (transient session-bus
    # error, /VirtualKeyboard interface unavailable, etc.), the user is left
    # with no IBus running and no Wayland IM bridge.  Best-effort fallback:
    # spin up ibus-daemon as a plain background process so basic IBus works.
    # The Wayland bridge won't be active without KWin spawning it, but that
    # recovers automatically at the next login.
    if not toggle_ok:
        run_command(["ibus-daemon", "-r", "-d"], quiet=True, check=False)


def reload_systemd_user(ctx: InstallContext) -> None:
    run_command(["systemctl", "--user", "daemon-reload"], quiet=True)
    run_command(["systemctl", "--user", "enable", SERVICE_NAME], quiet=True)
    run_command(["systemctl", "--user", "restart", SERVICE_NAME], quiet=True)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> int:
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

    gpu = _prompt_backend()
    if gpu:
        ctx = InstallContext(
            script_path=ctx.script_path, script_dir=ctx.script_dir,
            home=ctx.home, runtime_dir=ctx.runtime_dir, gpu=True,
        )

    print()
    preflight_ibus()
    for cmd in ("python3", "systemctl", "rsync", "dconf"):
        require_command(cmd)

    pkg = ctx.script_dir / "packaging"

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

    mode = "GPU + CPU fallback" if gpu else "CPU only"
    print(f"\n  \U0001f389 KDictate {__version__} installed ({mode})")
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
