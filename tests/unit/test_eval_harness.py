from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.harness import (
    align_words,
    format_alignment_lines,
    normalize_text,
    resolve_model_path,
    summarize_numeric,
)


class EvalHarnessTests(unittest.TestCase):
    def test_normalize_text_strips_punctuation_and_hyphens(self) -> None:
        self.assertEqual(normalize_text("Rock-'n'-roll, can't-stop."), "rock n roll cantstop")

    def test_align_words_reports_token_edits(self) -> None:
        alignment = align_words("alpha beta gamma", "alpha delta")

        self.assertEqual(alignment["counts"]["matches"], 1)
        self.assertEqual(alignment["counts"]["substitutions"], 1)
        self.assertEqual(alignment["counts"]["deletions"], 1)
        self.assertEqual(alignment["counts"]["insertions"], 0)
        self.assertAlmostEqual(alignment["wer"], 2 / 3)

    def test_format_alignment_lines_renders_side_by_side_rows(self) -> None:
        lines = format_alignment_lines(
            [
                {"op": "=", "reference": "alpha", "hypothesis": "alpha"},
                {"op": "~", "reference": "beta", "hypothesis": "delta"},
                {"op": "-", "reference": "gamma", "hypothesis": ""},
            ]
        )

        self.assertGreaterEqual(len(lines), 5)
        self.assertTrue(any("reference" in line and "hypothesis" in line for line in lines))
        self.assertTrue(any("beta" in line and "delta" in line for line in lines))

    def test_resolve_model_path_uses_installed_runtime_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "repo"
            installed_models_root = tmp_path / "installed" / "models"
            installed_model = installed_models_root / "whisper-large-v3-turbo-ct2"
            installed_model.mkdir(parents=True)

            resolved = resolve_model_path(
                "models/whisper-large-v3-turbo-ct2",
                project_root=project_root,
                installed_models_root=installed_models_root,
            )

            self.assertEqual(resolved, installed_model.resolve())

    def test_summarize_numeric_handles_empty_sequences(self) -> None:
        summary = summarize_numeric([])

        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["p90"])


if __name__ == "__main__":
    unittest.main()
