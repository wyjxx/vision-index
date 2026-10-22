"""Check current-model evaluation, test labels, and the standalone CLI."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from evaluation.evaluate import evaluate_model, main


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = {"queries": [{"query_id": "q", "query": "juicer"}],
                "products": [{"product_id": pid, "image_path": f"{pid}.png"} for pid in ("C", "A", "B")]}
        for split, pid in (("train", "A"), ("validation", "B"), ("test", "C")):
            rows[split] = [{"query_id": "q", "product_id": pid, "position": 1}]
        for name, values in rows.items():
            (self.root / f"{name}.jsonl").write_text("\n".join(map(json.dumps, values)), encoding="utf-8")
        for pid, color in (("A", "red"), ("B", "green"), ("C", "blue")):
            Image.new("RGB", (2, 2), color).save(self.root / f"{pid}.png")

        def processor(text=None, images=None, **kwargs):
            if text is not None:
                return {"input_ids": torch.tensor([[1.0, 0.0]])}
            vectors = {(255, 0, 0): [1.0, 0.0], (0, 128, 0): [0.0, 1.0], (0, 0, 255): [-1.0, 0.0]}
            return {"pixel_values": torch.tensor([vectors[image.getpixel((0, 0))] for image in images])}

        self.processor = processor
        self.model = SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1)]), dtype=torch.float32, eval=Mock(),
            get_image_features=Mock(side_effect=lambda pixel_values: SimpleNamespace(pooler_output=pixel_values)),
            get_text_features=lambda input_ids: SimpleNamespace(pooler_output=input_ids),
        )

    def evaluate_pair(self):
        baseline = self.root / "baseline"
        finetuned = self.root / "finetuned"
        evaluate_model(self.model, self.processor, self.root, baseline, "baseline", 2, split="test")
        self.assertEqual(self.model.get_image_features.call_count, 2)
        # Change image vectors so the test product C moves from rank 3 to rank 1.
        self.model.get_image_features = Mock(
            side_effect=lambda pixel_values: SimpleNamespace(pooler_output=-pixel_values))
        evaluate_model(self.model, self.processor, self.root, finetuned, "checkpoint", 2, split="test")
        self.assertEqual(self.model.get_image_features.call_count, 2)
        return baseline, finetuned

    def test_held_out_pairs_use_current_models_reencoded_images(self):
        baseline, finetuned = self.evaluate_pair()
        before = json.loads((baseline / "summary.json").read_text())
        after = json.loads((finetuned / "summary.json").read_text())
        self.assertAlmostEqual(before["test_pairs"]["ndcg@10"], 0.5)
        self.assertAlmostEqual(after["test_pairs"]["ndcg@10"], 1.0)
        self.assertEqual(before["test_pairs"]["pair_count"], 1)
        pair = json.loads((baseline / "pair_rankings.jsonl").read_text())
        self.assertEqual(pair["product_id"], "C")
        self.assertEqual(pair["model_rank"], 3)
        config = json.loads((finetuned / "config.json").read_text())
        self.assertEqual(config["label_splits"], ["train", "validation", "test"])

    def test_cli_loads_checkpoint_and_limits_queries_without_using_chroma(self):
        for name, row in (("queries", {"query_id": "q2", "query": "another query"}),
                          ("test", {"query_id": "q2", "product_id": "C", "position": 1})):
            with (self.root / f"{name}.jsonl").open("a", encoding="utf-8") as file:
                file.write("\n" + json.dumps(row))
        self.model.to = Mock(return_value=self.model)
        output = self.root / "cli"
        with patch("transformers.CLIPModel.from_pretrained", return_value=self.model) as loader, \
             patch("transformers.CLIPProcessor.from_pretrained", return_value=self.processor), \
             patch("torch.cuda.is_available", return_value=False), \
             patch.dict(sys.modules, {"app.storage.clip_vector_db": None, "app.ai.clip": None}):
            summary = main(["--checkpoint", "fixture_checkpoint", "--dataset-dir", str(self.root),
                            "--split", "test", "--limit", "1", "--output-dir", str(output)])
        self.assertEqual(loader.call_args.args[0], "fixture_checkpoint")
        self.assertEqual(loader.call_args.kwargs["dtype"], torch.float32)
        self.assertEqual(summary["queries"], 1)
        self.assertEqual(summary["candidates"], 3)
        self.assertEqual({path.name for path in output.iterdir()}, {
            "config.json", "summary.json", "pair_rankings.jsonl", "per_query_metrics.jsonl"})
        config = json.loads((output / "config.json").read_text())
        self.assertTrue(config["completed"])
        self.assertEqual(config["query_ids"], ["q"])


if __name__ == "__main__":
    unittest.main()
