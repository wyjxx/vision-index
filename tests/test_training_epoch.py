"""Check optimizer state and full-pair coverage across batches."""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import CLIPConfig, CLIPModel

from training.dataset import TrainingQuery
from training.epoch import train_one_epoch
from training.step import configure_projection_training


class TrainingEpochTests(unittest.TestCase):
    def test_epoch_reuses_adamw_state_and_trains_all_pairs(self):
        torch.manual_seed(42)
        config = CLIPConfig(
            text_config={"vocab_size": 20, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2,
                         "max_position_embeddings": 8, "bos_token_id": 0, "eos_token_id": 2},
            vision_config={"hidden_size": 16, "intermediate_size": 32,
                           "num_hidden_layers": 1, "num_attention_heads": 2,
                           "image_size": 32, "patch_size": 16},
            projection_dim=8,
        )
        model = CLIPModel(config)
        trainable = configure_projection_training(model)
        frozen_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                         if not parameter.requires_grad}
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, weight_decay=0.0)
        queries = [TrainingQuery(f"q{i}", str(i), [f"p{i}"]) for i in range(5)]
        def make_inputs(batch, dataset_dir, processor):
            count = len(batch.query_ids)
            return {"input_ids": torch.tensor([[0, 4 + i, 2] for i in range(count)]),
                    "attention_mask": torch.ones(count, 3, dtype=torch.long),
                    "pixel_values": torch.randn(len(batch.product_ids), 3, 32, 32)}

        # Replace image loading only; use the real model, loss, and optimizer.
        with patch("training.epoch.load_batch_inputs", side_effect=make_inputs) as loader:
            result = train_one_epoch(model, None, queries, Path("unused"), optimizer,
                                     queries_per_batch=2, images_per_query=1)

        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["trained_queries"], 5)
        self.assertEqual(result["trained_pairs"], 5)
        self.assertEqual(loader.call_count, 2)
        self.assertTrue(math.isfinite(result["mean_batch_loss"]))
        for parameter in trainable.values():
            self.assertEqual(optimizer.state[parameter]["step"].item(), 2)
        for name, parameter in model.named_parameters():
            if name in frozen_before:
                self.assertTrue(torch.equal(parameter, frozen_before[name]), name)


if __name__ == "__main__":
    unittest.main()
