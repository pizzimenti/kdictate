"""Entry point for the whisper-dictate IBus engine process."""

from __future__ import annotations

import logging
import sys
from typing import Sequence

import gi

gi.require_version("GLib", "2.0")
from gi.repository import GLib

from whisper_dictate.exceptions import IbusEngineError
from whisper_dictate.logging_utils import configure_logging
from whisper_dictate.ibus_engine.engine import ENGINE_NAME, build_engine_factory, load_ibus_module


def main(argv: Sequence[str] | None = None) -> int:
    """Run the IBus engine main loop."""

    del argv
    logger = configure_logging("whisper_dictate.ibus")
    logger.info("Starting IBus engine process for %s", ENGINE_NAME)

    try:
        ibus = load_ibus_module()
        ibus.init()
        factory = build_engine_factory()
    except IbusEngineError as exc:
        logger.error("IBus engine startup failed: %s", exc)
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected IBus engine startup failure")
        print(f"IBus engine startup failed: {exc}", file=sys.stderr)
        return 1

    loop = GLib.MainLoop()
    logger.info("IBus engine ready and entering GLib main loop")
    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("IBus engine interrupted")
    finally:
        del factory
        del ibus

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
