"""Check validation isolation, checkpoint reload, and model selection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel

from app.config import clip_model
from training.checkpoint import load_training_state
from training.run import main
from training.epoch import train_one_epoch


class TinyProcessor:
    """Convert fixture texts and images into small CLIP inputs."""

    def __init__(self):
        self.save_pretrained = Mock()

    def __call__(self, text=None, images=None, **kwargs):
        inputs = {}
        if text is not None:
            inputs["input_ids"] = torch.tensor([[0, 4 if value == "juicer" else 5, 2] for value in text])
            inputs["attention_mask"] = torch.ones(len(text), 3, dtype=torch.long)
        if images is not None:
            pixels = np.stack([np.asarray(image, dtype=np.float32) / 255 for image in images])
            inputs["pixel_values"] = torch.from_numpy(pixels).permute(0, 3, 1, 2)
        return inputs


class TrainingRunTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        rows = {
            "queries": [{"query_id": "q1", "query": "juicer"}, {"query_id": "q2", "query": "fridge"}],
            "products": [{"product_id": pid, "image_path": f"{pid}.png"} for pid in ("C", "B", "A")],
            "train": [{"query_id": "q1", "product_id": "A", "position": 1},
                      {"query_id": "q2", "product_id": "B", "position": 1}],
            "validation": [{"query_id": qid, "product_id": "C", "position": 1} for qid in ("q1", "q2")],
        }
        for name, values in rows.items():
            (self.dataset / f"{name}.jsonl").write_text("\n".join(map(json.dumps, values)), encoding="utf-8")
        (self.dataset / "test.jsonl").write_text("invalid JSON", encoding="utf-8")
        for pid, color in (("A", "red"), ("B", "blue"), ("C", "green")):
            Image.new("RGB", (32, 32), color).save(self.dataset / f"{pid}.png")
        config = CLIPConfig(
            text_config={"vocab_size": 20, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2,
                         "max_position_embeddings": 8, "bos_token_id": 0, "eos_token_id": 2},
            vision_config={"hidden_size": 16, "intermediate_size": 32,
                           "num_hidden_layers": 1, "num_attention_heads": 2,
                           "image_size": 32, "patch_size": 16},
            projection_dim=8,
        )
        self.model = CLIPModel(config)
        self.processor = TinyProcessor()

    def run_fixture(self, output, epochs=1, resume=None):
        model = CLIPModel.from_pretrained(resume, local_files_only=True) if resume else self.model
        arguments = ["--dataset-dir", str(self.dataset), "--output-dir", str(output),
                     "--epochs", str(epochs)]
        if resume:
            arguments.extend(["--resume", str(resume)])
        else:
            arguments.extend(["--queries-per-batch", "2", "--eval-batch-size", "2"])
        # Use a real small model; replace downloads and tokenization only.
        with patch("training.run.CLIPModel.from_pretrained", return_value=model) as model_loader, \
             patch("training.run.CLIPProcessor.from_pretrained", return_value=self.processor), \
             patch("torch.cuda.is_available", return_value=False):
            result = main(arguments)
            self.assertEqual(model_loader.call_count, 1)
            return result

    def test_full_run_saves_reloadable_checkpoint_without_test_labels(self):
        output = self.root / "run"
        result = self.run_fixture(output)
        self.assertTrue(result["completed"])
        epoch_result = result["history"][0]
        self.assertEqual(epoch_result["training"]["trained_pairs"], 2)
        for stage, summary in (("validation_before", result["validation_before"]),
                               ("validation_epoch_1", epoch_result["validation"])):
            config = json.loads((output / stage / "config.json").read_text())
            self.assertTrue(config["completed"])
            self.assertEqual(config["label_splits"], ["train", "validation"])
            self.assertEqual(config["candidate_count"], 3)
            self.assertEqual(summary["test_pairs"]["pair_count"], 2)
            self.assertEqual(summary["test_pairs"]["recall@100"], 1.0)
        checkpoint = output / "checkpoint_epoch_1"
        state = load_training_state(checkpoint)
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["run"]["history"][-1]["epoch"], 1)
        self.processor.save_pretrained.assert_called_once_with(checkpoint)
        reloaded = CLIPModel.from_pretrained(checkpoint, local_files_only=True)
        for name, parameter in self.model.state_dict().items():
            self.assertTrue(torch.equal(parameter, reloaded.state_dict()[name]), name)
        self.assertTrue(json.loads((output / "run.json").read_text())["completed"])

    def test_worse_validation_keeps_pretrained_as_best(self):
        summaries = [{"test_pairs": {"ndcg@10": score, "pair_count": 2}} for score in (0.7, 0.6)]
        with patch("training.run.evaluate_model", side_effect=summaries):
            result = self.run_fixture(self.root / "regression")
        self.assertEqual(result["best_epoch"], 0)
        self.assertEqual(result["best_model"], clip_model)
        self.assertAlmostEqual(result["history"][0]["held_out_metric_deltas"]["ndcg@10"], -0.1)

    def test_multiple_epochs_keep_optimizer_state_and_best_earlier_checkpoint(self):
        summaries = [{"test_pairs": {"ndcg@10": score, "pair_count": 2}}
                     for score in (0.5, 0.7, 0.6, 0.7)]
        output = self.root / "multiple_epochs"
        with patch("training.run.evaluate_model", side_effect=summaries), \
             patch("training.run.train_one_epoch", wraps=train_one_epoch) as train_epoch:
            result = self.run_fixture(output, epochs=3)

        calls = train_epoch.call_args_list
        self.assertEqual(len(calls), 3)
        optimizer = calls[0].args[4]
        for call, seed in zip(calls, (42, 43, 44)):
            self.assertIs(call.args[0], self.model)
            self.assertIs(call.args[4], optimizer)
            self.assertEqual(call.args[7], seed)
        for state in optimizer.state.values():
            self.assertEqual(state["step"].item(), 3)
        self.assertEqual([entry["epoch"] for entry in result["history"]], [1, 2, 3])
        self.assertTrue(all(entry["training"]["trained_pairs"] == 2 for entry in result["history"]))
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["best_model"], str((output / "checkpoint_epoch_1").resolve()))
        self.assertAlmostEqual(result["best_score"], 0.7)
        for epoch in (1, 2, 3):
            self.assertTrue((output / f"checkpoint_epoch_{epoch}" / "model.safetensors").is_file())
        saved = json.loads((output / "run.json").read_text())
        self.assertTrue(saved["completed"])
        self.assertEqual(len(saved["history"]), 3)

    def test_resume_matches_continuous_training_from_an_earlier_checkpoint(self):
        initial = {name: value.clone() for name, value in self.model.state_dict().items()}
        summaries = [{"test_pairs": {"ndcg@10": score, "pair_count": 2}}
                     for score in (0.5, 0.7, 0.6, 0.8)]
        continuous = self.root / "continuous"
        with patch("training.run.evaluate_model", side_effect=summaries):
            self.run_fixture(continuous, epochs=3)

        self.model.load_state_dict(initial)
        split = self.root / "split"
        with patch("training.run.evaluate_model", side_effect=summaries[:3]):
            self.run_fixture(split, epochs=2)

        # Resume epoch 1 even though the source run has already completed epoch 2.
        resumed = self.root / "resumed"
        with patch("training.run.evaluate_model", side_effect=summaries[2:]) as evaluate, \
             patch("training.run.train_one_epoch", wraps=train_one_epoch) as train_epoch:
            result = self.run_fixture(resumed, epochs=3, resume=split / "checkpoint_epoch_1")
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual([call.args[7] for call in train_epoch.call_args_list], [43, 44])
        self.assertEqual([entry["epoch"] for entry in result["history"]], [1, 2, 3])
        self.assertEqual(result["queries_per_batch"], 2)
        self.assertEqual(result["eval_batch_size"], 2)
        self.assertEqual(result["best_epoch"], 3)

        expected = CLIPModel.from_pretrained(continuous / "checkpoint_epoch_3", local_files_only=True)
        actual = CLIPModel.from_pretrained(resumed / "checkpoint_epoch_3", local_files_only=True)
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
        expected_state = load_training_state(continuous / "checkpoint_epoch_3")
        actual_state = load_training_state(resumed / "checkpoint_epoch_3")
        for key, values in expected_state["optimizer"]["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, actual_state["optimizer"]["state"][key][name]))
        self.assertTrue(torch.equal(expected_state["torch_rng_state"], actual_state["torch_rng_state"]))

    def test_resume_rejects_missing_optimizer_state(self):
        with self.assertRaises(SystemExit):
            main(["--resume", str(self.root / "old_checkpoint"), "--epochs", "5",
                  "--output-dir", str(self.root / "invalid")])
        self.assertFalse((self.root / "invalid").exists())

    def test_resume_rejects_changed_training_settings_and_dataset(self):
        summaries = [{"test_pairs": {"ndcg@10": score, "pair_count": 2}} for score in (0.5, 0.7)]
        source = self.root / "source"
        with patch("training.run.evaluate_model", side_effect=summaries):
            self.run_fixture(source)
        arguments = ["--resume", str(source / "checkpoint_epoch_1"), "--epochs", "2",
                     "--output-dir", str(self.root / "invalid")]
        with self.assertRaises(SystemExit):
            main(arguments + ["--queries-per-batch", "4"])
        with (self.dataset / "train.jsonl").open("a", encoding="utf-8") as file:
            file.write("\n")
        with self.assertRaises(SystemExit):
            main(arguments)
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()

