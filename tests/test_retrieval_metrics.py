"""Hand-calculated checks for V2 retrieval metrics; no models or databases."""

import unittest

from evaluation.metrics import ndcg_at_k, recall_at_k


class RetrievalMetricsTests(unittest.TestCase):
    def setUp(self):
        # A has gain 100, B has gain 50.
        self.labels = {"A": 1, "B": 51}

    def test_recall_uses_all_labels_as_denominator(self):
        self.assertEqual(recall_at_k(["X", "B", "A"], self.labels, 1), 0.0)
        self.assertEqual(recall_at_k(["X", "B", "A"], self.labels, 2), 0.5)
        self.assertEqual(recall_at_k(["X", "B", "A"], self.labels, 3), 1.0)

    def test_ndcg_preserves_positions_and_uses_linear_gains(self):
        # X contributes 0; B contributes 50/log2(3); A contributes 100/log2(4).
        # DCG = 81.54648767857287; ideal DCG = 131.54648767857287.
        expected = 81.54648767857287 / 131.54648767857287
        self.assertAlmostEqual(ndcg_at_k(["X", "B", "A"], self.labels, 3), expected)

    def test_perfect_and_reversed_order(self):
        self.assertAlmostEqual(ndcg_at_k(["A", "B"], self.labels, 2), 1.0)
        self.assertLess(ndcg_at_k(["B", "A"], self.labels, 2), 1.0)

    def test_top_k_and_missing_predictions(self):
        # At K=1, ideal gain is A's 100 even if only B was returned.
        self.assertAlmostEqual(ndcg_at_k(["B"], self.labels, 1), 0.5)
        self.assertAlmostEqual(ndcg_at_k(["B", "A"], self.labels, 1), 0.5)
        self.assertEqual(recall_at_k(["B"], self.labels, 10), 0.5)
        self.assertEqual(ndcg_at_k(["X", "A"], self.labels, 1), 0.0)

    def test_empty_or_unmatched_results(self):
        for metric in (recall_at_k, ndcg_at_k):
            self.assertEqual(metric([], self.labels, 5), 0.0)
            self.assertEqual(metric(["X"], self.labels, 5), 0.0)
            self.assertEqual(metric(["A"], {}, 5), 0.0)

    def test_invalid_inputs_are_rejected_without_changing_ranks(self):
        for metric in (recall_at_k, ndcg_at_k):
            for k in (0, -1, True, 1.5):
                with self.subTest(metric=metric.__name__, k=k), self.assertRaises(ValueError):
                    metric(["A"], self.labels, k)
            with self.assertRaises(ValueError):
                metric(["A", "A"], self.labels, 5)
            for position in (0, 101, True, 1.5):
                with self.subTest(position=position), self.assertRaises(ValueError):
                    metric(["A"], {"A": position}, 5)


if __name__ == "__main__":
    unittest.main()
