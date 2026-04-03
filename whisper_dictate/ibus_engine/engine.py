"""Runtime IBus engine wiring for whisper-dictate."""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Callable

from whisper_dictate.constants import DBUS_BUS_NAME, DBUS_INTERFACE
from whisper_dictate.exceptions import IbusEngineError
from whisper_dictate.logging_utils import configure_logging
from whisper_dictate.ibus_engine.controller import DictationEngineController, EngineAdapter
from whisper_dictate.ibus_engine.dbus_client import DaemonSignalBridge

ENGINE_NAME = DBUS_INTERFACE
ENGINE_OBJECT_PATH = "/io/github/pizzimenti/WhisperDictate1/engine"
LOGGER_NAME = "whisper_dictate.ibus"


def load_ibus_module() -> ModuleType:
    """Load the IBus typelib lazily so tests can import this package without it."""

    import gi

    gi.require_version("IBus", "1.0")
    from gi.repository import IBus  # type: ignore[import-not-found]

    return IBus


class _IbusRenderAdapter:
    """Translate controller render operations into IBus API calls."""

    def __init__(self, engine: Any, ibus_module: ModuleType) -> None:
        self._engine = engine
        self._ibus = ibus_module

    def update_preedit(self, text: str, *, visible: bool, focus_mode: str) -> None:
        ibus_text = self._ibus.Text.new_from_string(text)
        mode = (
            self._ibus.PreeditFocusMode.COMMIT
            if focus_mode == "commit"
            else self._ibus.PreeditFocusMode.CLEAR
        )
        self._engine.update_preedit_text_with_mode(ibus_text, len(text), visible, mode)
        if visible:
            self._engine.show_preedit_text()
        else:
            self._engine.hide_preedit_text()

    def commit_text(self, text: str) -> None:
        self._engine.commit_text(self._ibus.Text.new_from_string(text))


def create_ibus_engine_class(ibus_module: ModuleType | None = None) -> type[Any]:
    """Create the concrete IBus.Engine subclass used by ibus-daemon."""

    ibus = ibus_module or load_ibus_module()
    logger = configure_logging(LOGGER_NAME)

    class WhisperDictateEngine(ibus.Engine):  # type: ignore[misc,valid-type]
        """IBus engine that mirrors daemon transcripts into preedit/commit."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._logger = logger.getChild("engine")
            self._adapter = _IbusRenderAdapter(self, ibus)
            self._controller = DictationEngineController(self._adapter, self._logger)
            self._bridge = DaemonSignalBridge(self._controller, self._logger)
            self._bridge.start()
            self._logger.info(
                "IBus engine initialized for daemon bus %s at object path %s",
                DBUS_BUS_NAME,
                ENGINE_OBJECT_PATH,
            )

        def do_enable(self) -> None:
            self._controller.enable()

        def do_disable(self) -> None:
            self._controller.disable()

        def do_focus_in(self) -> None:
            self._controller.focus_in()

        def do_focus_out(self) -> None:
            self._controller.focus_out()

        def do_reset(self) -> None:
            self._controller.reset()

        def do_set_surrounding_text(self, text: Any, cursor_pos: int, anchor_pos: int) -> None:
            self._controller.set_surrounding_text(_coerce_text(text), cursor_pos, anchor_pos)

        def do_destroy(self) -> None:
            self._bridge.stop()
            self._logger.info("IBus engine destroyed")
            try:
                super().do_destroy()
            except Exception:  # noqa: BLE001
                # Some IBus builds do not expose a parent do_destroy implementation.
                pass

    WhisperDictateEngine.__name__ = "WhisperDictateEngine"
    WhisperDictateEngine.__qualname__ = "WhisperDictateEngine"
    return WhisperDictateEngine


def create_engine_instance(engine_name: str = ENGINE_NAME, object_path: str = ENGINE_OBJECT_PATH) -> Any:
    """Create a concrete engine instance for the IBus factory."""

    ibus = load_ibus_module()
    engine_type = create_ibus_engine_class(ibus)
    try:
        bus = ibus.Bus.new()
        connection = bus.get_connection()
        return ibus.Engine.new_with_type(engine_type.__gtype__, engine_name, object_path, connection)
    except Exception as exc:  # noqa: BLE001
        raise IbusEngineError(f"Unable to create IBus engine instance: {exc}") from exc


def build_engine_factory() -> Any:
    """Build an IBus factory that can construct the whisper-dictate engine."""

    ibus = load_ibus_module()
    engine_type = create_ibus_engine_class(ibus)
    bus = ibus.Bus.new()
    factory = ibus.Factory.new(bus.get_connection())
    factory.add_engine(ENGINE_NAME, engine_type.__gtype__)
    return factory


def _coerce_text(text: Any) -> str:
    """Convert an IBus text object or plain string into a string value."""

    getter = getattr(text, "get_text", None)
    if callable(getter):
        return str(getter())
    return str(text)
