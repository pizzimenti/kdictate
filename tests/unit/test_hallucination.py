"""Tests for Whisper hallucination detection and postprocess_transcript."""

from __future__ import annotations

import unittest

from kdictate.audio_common import (
    is_hallucination,
    postprocess_transcript,
)


class IsHallucinationTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertTrue(is_hallucination("Thank you"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(is_hallucination("THANK YOU"))
        self.assertTrue(is_hallucination("Bye"))

    def test_strips_punctuation(self) -> None:
        self.assertTrue(is_hallucination("Thank you."))
        self.assertTrue(is_hallucination("Okay!"))
        self.assertTrue(is_hallucination('"you"'))

    def test_collapses_internal_whitespace(self) -> None:
        self.assertTrue(is_hallucination("thank   you"))
        self.assertTrue(is_hallucination("the\t end"))

    def test_real_sentence_not_filtered(self) -> None:
        self.assertFalse(is_hallucination("Thank you for your help"))

    def test_empty_string(self) -> None:
        self.assertFalse(is_hallucination(""))

    def test_substring_not_matched(self) -> None:
        self.assertFalse(is_hallucination("I said thank you to him"))

    def test_all_phrases_detected(self) -> None:
        for phrase in (
            "thank you", "thanks for watching", "thank you for watching",
            "you", "bye", "goodbye", "the end", "thanks", "so", "okay",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_hallucination(phrase))


class PostprocessTranscriptTest(unittest.TestCase):
    def test_suppresses_hallucination(self) -> None:
        self.assertEqual(postprocess_transcript("Thank you."), "")

    def test_suppresses_all_phrases(self) -> None:
        for phrase in ("you", "Bye.", "Okay!", "So", "Thanks"):
            with self.subTest(phrase=phrase):
                self.assertEqual(postprocess_transcript(phrase), "")

    def test_normalizes_whitespace(self) -> None:
        self.assertEqual(postprocess_transcript("  hello\n world  "), "hello world")

    def test_empty_input(self) -> None:
        self.assertEqual(postprocess_transcript(""), "")

    def test_real_sentence_passes(self) -> None:
        text = "Please send the report"
        self.assertEqual(postprocess_transcript(text), text)

    def test_sentence_containing_hallucination_phrase_passes(self) -> None:
        self.assertEqual(
            postprocess_transcript("Thank you for your help"),
            "Thank you for your help",
        )
