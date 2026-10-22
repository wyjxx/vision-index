"""Exact ranking checks with small, hand-calculated vectors."""

import unittest

import numpy as np

from evaluation.evaluate import rank_products


class ExactRankingTests(unittest.TestCase):
    def test_all_candidates_and_cosine_normalization(self):
        images = np.array([[0, 3], [-2, 0], [4, 0]], dtype=np.float32)
        original = images.copy()
        results = rank_products([2, 0], ["B", "C", "A"], images)
        self.assertEqual([r["product_id"] for r in results], ["A", "B", "C"])
        self.assertEqual([r["model_rank"] for r in results], [1, 2, 3])
        self.assertEqual([r["similarity"] for r in results], [1.0, 0.0, -1.0])
        np.testing.assert_array_equal(images, original)

    def test_ties_use_product_id_independently_of_input_order(self):
        results = rank_products([1, 0], ["B", "A"], [[1, 0], [2, 0]])
        self.assertEqual([r["product_id"] for r in results], ["A", "B"])

    def test_invalid_vectors_and_ids_are_rejected(self):
        cases = [
            ([0, 0], ["A"], [[1, 0]]),
            ([1, 0], ["A"], [[0, 0]]),
            ([1, 0], ["A"], [[1, float("nan")]]),
            ([1, 0], ["A"], [[1, 0, 0]]),
            ([1, 0], ["A", "A"], [[1, 0], [0, 1]]),
            ([1, 0], [""], [[1, 0]]),
            ([1, 0], [], []),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                rank_products(*arguments)


if __name__ == "__main__":
    unittest.main()
