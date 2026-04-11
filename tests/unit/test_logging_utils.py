"""Tests for the shared logging helper."""

from __future__ import annotations

import io
import logging
import unittest

from kdictate.logging_utils import configure_logging, get_propagating_child


class LoggingUtilsTests(unittest.TestCase):
    """Verify logger setup is deterministic and does not duplicate handlers."""

    def test_configure_logging_installs_one_handler(self) -> None:
        stream = io.StringIO()
        logger_name = "kdictate.tests.logging_utils"
        logger = logging.getLogger(logger_name)
        original_handlers = list(logger.handlers)
        for handler in original_handlers:
            logger.removeHandler(handler)

        try:
            configured = configure_logging(logger_name, stream=stream)
            self.assertIs(configured, logger)
            self.assertEqual(len(configured.handlers), 1)
            self.assertFalse(configured.propagate)

            configured.info("hello world")
            output = stream.getvalue()
            self.assertIn(logger_name, output)
            self.assertIn("hello world", output)

            configure_logging(logger_name, stream=stream)
            self.assertEqual(len(configured.handlers), 1)
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            for handler in original_handlers:
                logger.addHandler(handler)


class GetPropagatingChildTests(unittest.TestCase):
    """Verify parent-with-children pattern funnels into one handler."""

    def setUp(self) -> None:
        self._parent_name = "kdictate.tests.propagating_parent"
        self._child_suffixes = ("alpha", "beta", "gamma")
        self._touched: list[logging.Logger] = []

    def tearDown(self) -> None:
        for logger in self._touched:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            logger.propagate = True

    def _track(self, logger: logging.Logger) -> logging.Logger:
        self._touched.append(logger)
        return logger

    def test_child_has_no_handlers_and_propagates(self) -> None:
        parent = self._track(logging.getLogger(self._parent_name))
        child = self._track(get_propagating_child(parent, "alpha"))

        self.assertEqual(child.name, f"{self._parent_name}.alpha")
        self.assertEqual(child.handlers, [])
        self.assertTrue(child.propagate)

    def test_existing_child_handlers_are_cleared(self) -> None:
        parent = self._track(logging.getLogger(self._parent_name))
        pre_existing = self._track(logging.getLogger(f"{self._parent_name}.beta"))
        pre_existing.addHandler(logging.NullHandler())
        pre_existing.propagate = False
        self.assertEqual(len(pre_existing.handlers), 1)

        reset = get_propagating_child(parent, "beta")

        self.assertIs(reset, pre_existing)
        self.assertEqual(reset.handlers, [])
        self.assertTrue(reset.propagate)

    def test_one_handler_receives_messages_from_all_children(self) -> None:
        # Set up a parent with a single capturing stream handler, and three
        # propagating children. Each child logs once. The capturing handler
        # should see exactly three records — proving messages funnel through
        # one handler instead of being duplicated by per-child handlers.
        parent = self._track(logging.getLogger(self._parent_name))
        for handler in list(parent.handlers):
            parent.removeHandler(handler)
        stream = io.StringIO()
        capture = logging.StreamHandler(stream)
        capture.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        parent.addHandler(capture)
        parent.setLevel(logging.INFO)
        parent.propagate = False

        children = [
            self._track(get_propagating_child(parent, suffix))
            for suffix in self._child_suffixes
        ]
        for child, suffix in zip(children, self._child_suffixes):
            child.setLevel(logging.INFO)
            child.info("hello from %s", suffix)

        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 3)
        for line, suffix in zip(lines, self._child_suffixes):
            self.assertIn(f"{self._parent_name}.{suffix}", line)
            self.assertIn(f"hello from {suffix}", line)
