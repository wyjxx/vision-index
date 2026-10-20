"""Check that compact output retains far-away labels and full-ranking scores."""

import io
import json
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
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

    def test_ranking_entry_point_writes_compact_artifacts(self):
        from evaluation.ranking import main

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "dataset"
            directory.mkdir()
            rows = {"queries": [{"query_id": "q", "query": "test"}],
                    "products": [{"product_id": pid} for pid in ("A", "B", "C")]}
            for split, pid in (("train", "A"), ("validation", "B"), ("test", "C")):
                rows[split] = [{"query_id": "q", "product_id": pid, "position": 1}]
            for name, values in rows.items():
                (directory / f"{name}.jsonl").write_text("\n".join(map(json.dumps, values)), encoding="utf-8")
            output = Path(temporary) / "output"
            encoder = Mock(return_value=[[1, 0]])
            storage = Mock(return_value=[[1, 0], [0, 1], [-1, 0]])
            modules = {"app.ai.clip": SimpleNamespace(encode_texts=encoder),
                       "app.storage.clip_vector_db": SimpleNamespace(get_clip_embeddings=storage)}
            with patch.dict(sys.modules, modules):
                main(["--dataset-dir", str(directory), "--split", "test", "--output-dir", str(output)])
            self.assertEqual({p.name for p in output.iterdir()}, {
                "pair_rankings.jsonl", "per_query_metrics.jsonl", "summary.json", "config.json"})
            self.assertTrue(json.loads((output / "config.json").read_text())["completed"])
            pair = json.loads((output / "pair_rankings.jsonl").read_text())
            self.assertEqual(pair["product_id"], "C")
            self.assertEqual(pair["model_rank"], 3)
            summary = json.loads((output / "summary.json").read_text())
            self.assertAlmostEqual(summary["test_pairs"]["ndcg@10"], 0.5)
            storage.assert_called_once_with(["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
