"""Check multiple positives, shared products, and held-out label isolation."""

import json
from collections import Counter
import tempfile
import unittest
from pathlib import Path

from training.dataset import TrainingQuery, iter_training_batches, load_training_queries


class TrainingDatasetTests(unittest.TestCase):
    def test_shared_product_is_one_column_with_multiple_positive_queries(self):
        queries = [TrainingQuery("q1", "juicer", ["A", "B"]),
                   TrainingQuery("q2", "kitchen appliance", ["B", "C"])]
        batch = next(iter_training_batches(queries, 2, 2))
        self.assertEqual(len(batch.product_ids), 3)
        for qid, row in zip(batch.query_ids, batch.positive_mask):
            positives = {pid for pid, value in zip(batch.product_ids, row) if value}
            self.assertEqual(positives, {"A", "B"} if qid == "q1" else {"B", "C"})

    def test_mask_recognizes_positives_sampled_by_other_queries(self):
        # q1 samples one image, but other queries bring both A and B into the batch.
        queries = [TrainingQuery("q1", "appliance", ["A", "B"]),
                   TrainingQuery("q2", "juicer", ["A"]),
                   TrainingQuery("q3", "fridge", ["B"])]
        batch = next(iter_training_batches(queries, 3, 1))
        row = batch.positive_mask[batch.query_ids.index("q1")]
        self.assertEqual(row, [True, True])

    def test_training_loader_never_uses_held_out_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = {
                "queries": [{"query_id": "q", "query": "juicer"}],
                "train": [{"query_id": "q", "product_id": "A", "position": 10}],
            }
            for name, values in rows.items():
                (directory / f"{name}.jsonl").write_text("\n".join(map(json.dumps, values)), encoding="utf-8")
            for name in ("validation", "test"):
                (directory / f"{name}.jsonl").write_text("invalid JSON", encoding="utf-8")
            self.assertEqual(load_training_queries(directory), [TrainingQuery("q", "juicer", ["A"])])

    def test_seed_is_repeatable_and_single_query_remainder_is_merged(self):
        queries = [TrainingQuery(f"q{i}", str(i), [f"p{i}"]) for i in range(5)]
        first = list(iter_training_batches(queries, 2, 1, seed=7))
        self.assertEqual(first, list(iter_training_batches(queries, 2, 1, seed=7)))
        self.assertEqual([len(batch.query_ids) for batch in first], [2, 3])
        self.assertEqual({qid for batch in first for qid in batch.query_ids}, {q.query_id for q in queries})

    def test_all_pairs_are_scheduled_once_with_uneven_query_sizes(self):
        queries = [TrainingQuery("q1", "appliance", ["A", "B", "C", "D", "E", "F", "G"]),
                   TrainingQuery("q2", "juicer", ["A", "H"]),
                   TrainingQuery("q3", "fridge", ["I"])]
        expected = Counter((query.query_id, pid) for query in queries for pid in query.product_ids)
        for seed in (0, 7, 42):
            with self.subTest(seed=seed):
                batches = list(iter_training_batches(queries, 2, 2, seed))
                actual = Counter(pair for batch in batches for pair in batch.pairs)
                self.assertEqual(actual, expected)
                for batch in batches:
                    self.assertGreaterEqual(len(batch.query_ids), 2)
                    self.assertEqual(len(set(batch.product_ids)), len(batch.product_ids))
                    for qid, pid in batch.pairs:
                        self.assertTrue(batch.positive_mask[batch.query_ids.index(qid)][batch.product_ids.index(pid)])

    def test_single_query_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_training_batches([TrainingQuery("q", "juicer", ["A", "B"])]))


if __name__ == "__main__":
    unittest.main()
