"""Tests for Whisper hallucination detection and postprocess_transcript."""

from __future__ import annotations

import unittest

import numpy as np

from kdictate.audio_common import (
    HALLUCINATION_RMS_CEILING,
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
    def _silent_chunks(self) -> list[np.ndarray]:
        """PCM chunks with avg RMS well below the ceiling."""
        return [np.zeros(1600, dtype=np.int16)]

    def _loud_chunks(self) -> list[np.ndarray]:
        """PCM chunks with avg RMS well above the ceiling."""
        rms_target = HALLUCINATION_RMS_CEILING * 3
        samples = np.full(1600, rms_target, dtype=np.int16)
        return [samples]

    def test_suppresses_hallucination_on_silence(self) -> None:
        self.assertEqual(postprocess_transcript("Thank you.", self._silent_chunks()), "")

    def test_allows_hallucination_phrase_on_loud_audio(self) -> None:
        self.assertEqual(
            postprocess_transcript("Thank you.", self._loud_chunks()),
            "Thank you.",
        )

    def test_normalizes_whitespace(self) -> None:
        self.assertEqual(
            postprocess_transcript("  hello\n world  ", self._loud_chunks()),
            "hello world",
        )

    def test_empty_input(self) -> None:
        self.assertEqual(postprocess_transcript("", self._loud_chunks()), "")

    def test_real_sentence_passes(self) -> None:
        text = "Please send the report"
        self.assertEqual(postprocess_transcript(text, self._silent_chunks()), text)
