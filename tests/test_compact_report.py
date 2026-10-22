"""Check that compact output retains far-away labels and full-ranking scores."""

import io
import json
import tempfile
import unittest
from math import log2
from pathlib import Path

from evaluation.report import load_known_labels, write_query_outputs, build_summary


class CompactReportTests(unittest.TestCase):
    def test_deep_pair_keeps_its_rank_and_full_ndcg_contribution(self):
        results = [{"product_id": f"P{i:03}", "similarity": 1 - i / 105, "model_rank": i + 1}
                   for i in range(105)]
        query = {"query_id": "q", "query": "test", "relevant_products": {"P001": 1, "P104": 51}}
        known = {**query["relevant_products"], "P000": 21}
        pairs, metrics = io.StringIO(), io.StringIO()
        report = write_query_outputs(query, results, known, pairs, metrics)
        saved_pairs = [json.loads(line) for line in pairs.getvalue().splitlines()]
        self.assertEqual([r["model_rank"] for r in saved_pairs], [2, 105])
        self.assertEqual([r["reference_position"] for r in saved_pairs], [1, 51])
        expected_ndcg10 = (100 / log2(3)) / (100 + 50 / log2(3))
        self.assertAlmostEqual(report["test_pairs"]["ndcg@10"], expected_ndcg10)
        self.assertGreater(report["test_pairs"]["ndcg@105"], report["test_pairs"]["ndcg@100"])
        self.assertEqual(json.loads(metrics.getvalue()), report)
        summary = build_summary([report], {"model": "fixture", "split": "test", "candidate_count": 105})
        self.assertEqual(summary["test_pairs"]["pair_count"], 2)
        self.assertEqual(summary["test_pairs"]["recall@100"], 0.5)

    def test_validation_does_not_read_test_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, pid in (("train", "A"), ("validation", "B")):
                row = {"query_id": "q", "product_id": pid, "position": 1}
                (directory / f"{name}.jsonl").write_text(json.dumps(row), encoding="utf-8")
            (directory / "test.jsonl").write_text("invalid JSON", encoding="utf-8")
            self.assertEqual(load_known_labels(directory, "validation"), {"q": {"A": 1, "B": 1}})

if __name__ == "__main__":
    unittest.main()
