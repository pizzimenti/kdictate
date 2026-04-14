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

- **gdbus toggle `VirtualKeyboard.enabled` false → true** — only restarts a process
  that is already registered with KWin via the Wayland IM protocol.  If no such
  process is running the toggle is a no-op (available stays false).

- **`qdbus6 org.kde.KWin /KWin reconfigure` alone** — causes KWin to re-read kwinrc
  (so it now knows the InputMethod desktop file path), but does not spawn the
  input method process.

- **`ibus-ui-gtk3 --enable-wayland-im` launched from a terminal** — crashes: the
  compositor Wayland socket with `zwp_input_method_v2` is not available to
  non-KWin-spawned processes.

### Working hot-start sequence (used by install.py)

```
# 1. Tell KWin to reload kwinrc so it knows which InputMethod desktop file to use.
qdbus6 org.kde.KWin /KWin reconfigure

# 2. Bootstrap ibus-daemon so a process is registered on the session D-Bus.
#    -r = replace any stale instance; -d = daemonise.
ibus-daemon -r -d --panel disable
sleep 1

# 3. "ibus restart" sends Exit() to the running daemon.
#    KWin detects the process death and re-launches via the InputMethod desktop
#    file, giving the new process the correct Wayland socket.
#    ibus-ui-gtk3 --enable-wayland-im registers with the compositor →
#    VirtualKeyboard.available becomes true.
ibus restart
```

Step 2 is necessary because `ibus restart` calls `ibus exit` on a running daemon;
if no daemon is running the exit call fails and the restart does nothing.  The
bootstrapped daemon does not need Wayland IM support — it only exists long enough
for KWin to observe its death and perform the proper restart.
