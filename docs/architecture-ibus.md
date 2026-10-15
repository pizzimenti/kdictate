# IBus-only architecture

`kdictate` is structured around a strict split:

- the core daemon owns audio capture, VAD, transcription, runtime state, and a session D-Bus API
- the IBus engine is the only component allowed to place text into applications

The daemon publishes:

- `StateChanged(state)`
- `PartialTranscript(text)`
- `FinalTranscript(text)`
- `ErrorOccurred(code, message)`

The IBus engine consumes those events and maps them to:

- partial transcript -> IBus preedit
- final transcript -> IBus commit

No synthetic typing, clipboard paste fallback, or mixed insertion backend is allowed in the redesign.

---

## KDE Plasma Wayland: IBus startup and hot-start

### How IBus starts at session login

On KDE Plasma Wayland, `ibus-daemon` is **not** started directly by the user or by
an autostart entry.  Instead, KWin manages the input method lifecycle:

1. KWin reads `~/.config/kwinrc` at startup and finds:
   ```ini
   [Wayland]
   InputMethod=/usr/share/applications/org.freedesktop.IBus.Panel.Wayland.Gtk3.desktop
   VirtualKeyboardEnabled=true
   ```
2. KWin spawns the desktop-file `Exec` line with a dedicated Wayland socket that
   exposes the `zwp_input_method_v2` protocol:
   ```
   /usr/lib/ibus/ibus-ui-gtk3 --enable-wayland-im \
       --exec-daemon --daemon-args "--xim --panel disable"
   ```
3. `ibus-ui-gtk3 --enable-wayland-im` registers with the compositor; KWin sets
   `VirtualKeyboard.available = true`.
4. `--exec-daemon` starts `ibus-daemon --xim --panel disable` as a child.
5. `ibus-daemon` auto-spawns preloaded engines (including `ibus-engine-kdictate`).

Processes started outside KWin's supervision (e.g. from a terminal) do **not** receive
that dedicated socket, so `ibus-ui-gtk3 --enable-wayland-im` exits immediately with
"No input_method global" and may segfault.

### Why the installer needs a hot-start

`install.py` writes the `InputMethod` and `VirtualKeyboardEnabled` keys via
`kwriteconfig6` during an already-running session.  KWin has the old (often null)
`InputMethod` value in memory; it does not re-read `kwinrc` automatically.  The IBus
panel is therefore never started until the next logout/login — unless a hot-start
sequence is run at the end of install.

### What does NOT work for hot-start

- **`qdbus6 org.kde.KWin /KWin reconfigure` alone** — causes KWin to re-read kwinrc
  (so it now knows the InputMethod desktop file path), but does not spawn the
  input method process.

- **`gdbus`/`qdbus` toggle of `VirtualKeyboard.enabled`** — the property no
  longer exists.  Plasma 6.7's `org.kde.kwin.VirtualKeyboard` interface exposes
  `available`/`active`/`visible`/`mode`, none of which relaunch the InputMethod.
  Installers ≤0.16.0 relied on this toggle; when Plasma removed the property the
  toggle failed *after* the daemon had already been killed, the fallback started
  a bare `ibus-daemon -r -d` with no Wayland IM bridge, and every install broke
  typing until relogin (root-caused 2026-08-10).

- **`ibus restart`** — sends `Exit(restart=true)`, which causes ibus-daemon to
  re-exec itself in place with the same args.  KWin sees no D-Bus name owner
  change, so it never invokes the InputMethod desktop file and the Wayland IM
  bridge is never launched.  The daemon stays up but with `--panel disable` (or
  whatever args it was started with), and `VirtualKeyboard.available` stays false.

- **`ibus-ui-gtk3 --enable-wayland-im` launched from a terminal** — crashes: the
  compositor Wayland socket with `zwp_input_method_v2` is not available to
  non-KWin-spawned processes.

### Working relaunch sequence (used by install.py ≥0.17.0)

Traced in KWin 6.7.4 source (`main_wayland.cpp` `refreshSettings`,
`inputmethod.cpp` `setInputMethodCommand`, `workspace.cpp`
`slotReconfigure`): KWin watches kwinrc with a **KConfigWatcher**, which only
fires on **notified** config writes (`kwriteconfig6 --notify`, the same
KConfig::Notify mechanism the Virtual Keyboard KCM uses). A notified change
to `[Wayland] InputMethod` stops the old IM process, creates a fresh private
Wayland connection, exports its FD as `WAYLAND_SOCKET`, and launches the
desktop file's Exec.

Three facts that make or break the sequence:

- **A plain (non-notified) `kwriteconfig6` write does nothing until next
  login.** The file changes; KConfigWatcher never fires. This exact silence
  was the failed first attempt at a live repair.
- **`qdbus6 org.kde.KWin /KWin reconfigure` is not part of this path.**
  `Workspace::slotReconfigure()` never touches the input method.
- **Writing the same value is a no-op** — KWin early-returns when the new
  Exec equals its current command — so restarting a dead IM with an
  unchanged config requires the notified delete → notified restore pair.

```sh
# 0. Only when the bridge is already missing!  A healthy session must never
#    have its ibus-daemon killed — engines respawn on demand, so upgrades
#    need only `ibus write-cache` + killing kdictate engine processes.

# 1. Notified delete: KWin clears its input-method command.
kwriteconfig6 --notify --file ~/.config/kwinrc \
    --group Wayland --key InputMethod --delete
sleep 1

# 2. Clear the bridgeless daemon so the relaunched bridge's --exec-daemon
#    child does not collide with it (only PIDs proven to be this session's).
pkill -x ibus-daemon
sleep 0.5

# 3. Notified restore: the changed value makes KWin launch the full stack.
kwriteconfig6 --notify --file ~/.config/kwinrc \
    --group Wayland --key InputMethod \
    /usr/share/applications/org.freedesktop.IBus.Panel.Wayland.Gtk3.desktop

# 4. Verify: bridge AND its ibus-daemon child must both appear.
pgrep -af 'ibus-ui-gtk3.*--enable-wayland-im'
pgrep -x ibus-daemon
```

Notes: `/VirtualKeyboard`'s D-Bus surface (`mode`, `active`, `visible`,
`available`, `forceActivate()`) contains **no** method that relaunches the IM
process, and `available=true` only means the command string is nonempty — it
does not prove the process is alive. `VirtualKeyboardEnabled`/`Mode` gate
whether KWin treats the IM as enabled, not whether the process is spawned.
If `kwriteconfig6` is unavailable there is no in-session recovery: the
setting takes effect at the next login.
