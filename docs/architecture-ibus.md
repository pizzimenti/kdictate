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

- **`gdbus`/`qdbus` toggle `VirtualKeyboard.enabled` false → true alone, immediately
  after install** — KWin still has the old (often null) `InputMethod` in memory
  because `kwriteconfig6` only wrote to disk, so the toggle has nothing to launch.
  The toggle becomes effective once KWin has been told to re-read kwinrc.

- **`ibus restart`** — sends `Exit(restart=true)`, which causes ibus-daemon to
  re-exec itself in place with the same args.  KWin sees no D-Bus name owner
  change, so it never invokes the InputMethod desktop file and the Wayland IM
  bridge is never launched.  The daemon stays up but with `--panel disable` (or
  whatever args it was started with), and `VirtualKeyboard.available` stays false.

- **`ibus-ui-gtk3 --enable-wayland-im` launched from a terminal** — crashes: the
  compositor Wayland socket with `zwp_input_method_v2` is not available to
  non-KWin-spawned processes.

### Working hot-start sequence (used by install.py)

```sh
# 1. Tell KWin to reload kwinrc so it picks up the InputMethod desktop file
#    that kwriteconfig6 just wrote.
qdbus6 org.kde.KWin /KWin reconfigure

# 2. Kill any stale ibus-daemon so the toggle below has a clean slate.  Without
#    this, a daemon already registered on the session bus (especially one
#    started with --panel disable) makes the toggle a no-op — KWin sees the
#    name still owned and doesn't cold-start the InputMethod desktop file.
pkill -x ibus-daemon
sleep 0.5

# 3. Toggle KWin's VirtualKeyboard.enabled from false to true.  This is the
#    signal that makes KWin invoke the InputMethod desktop file:
#       ibus-ui-gtk3 --enable-wayland-im --exec-daemon …
#    which spawns both the daemon and the Wayland IM bridge in one shot,
#    registers the input method with the compositor, and sets
#    VirtualKeyboard.available=true.
qdbus6 --literal org.kde.KWin /VirtualKeyboard \
    org.freedesktop.DBus.Properties.Set \
    org.kde.kwin.VirtualKeyboard enabled false
sleep 0.5
qdbus6 --literal org.kde.KWin /VirtualKeyboard \
    org.freedesktop.DBus.Properties.Set \
    org.kde.kwin.VirtualKeyboard enabled true
```

The full sequence is required — the `reconfigure` makes the toggle meaningful,
and the `pkill` makes it cold-start instead of no-op.  If neither `qdbus6` nor
`qdbus` is available on the host, there is no in-session recovery: the new
`InputMethod` setting takes effect at the next login when KWin re-reads kwinrc
during startup.
