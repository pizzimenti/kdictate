import unittest

from benchmarks.run import edit_distance, words


class BenchmarkMetricsTests(unittest.TestCase):
    def test_case_and_punctuation_do_not_count_as_word_errors(self):
        self.assertEqual(words("It's working, Bradley!"), words("IT’S WORKING BRADLEY"))

    def test_substitution_insertion_and_deletion(self):
        reference = words("the quick brown fox")
        self.assertEqual(edit_distance(reference, words("the slow brown fox")), 1)
        self.assertEqual(edit_distance(reference, words("the quick brown fox jumps")), 1)
        self.assertEqual(edit_distance(reference, words("the quick fox")), 1)
        self.assertEqual(edit_distance(reference, []), 4)
        self.assertEqual(edit_distance([], words("unexpected words")), 2)
