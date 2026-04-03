"""IBus frontend for whisper-dictate."""

from .controller import DictationEngineController, EngineState
from .engine import create_ibus_engine_class

__all__ = [
    "DictationEngineController",
    "EngineState",
    "create_ibus_engine_class",
]
