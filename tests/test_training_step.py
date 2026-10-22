"""Check projection updates and frozen weights with a small CLIP model."""

import unittest

import torch
from transformers import CLIPConfig, CLIPModel

from training.step import configure_projection_training, train_one_step


class TrainingStepTests(unittest.TestCase):
    def test_one_step_updates_only_projections(self):
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
        self.assertEqual(set(trainable), {"text_projection.weight", "visual_projection.weight"})
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        inputs = {"input_ids": torch.tensor([[0, 4, 2], [0, 5, 2]]),
                  "attention_mask": torch.ones(2, 3, dtype=torch.long),
                  "pixel_values": torch.randn(3, 3, 32, 32)}
        mask = torch.tensor([[True, True, False], [False, False, True]])
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, weight_decay=0.0)
        result = train_one_step(model, inputs, mask, optimizer)

        self.assertLess(result["loss_after"], result["loss_before"])
        for name, parameter in model.named_parameters():
            if name in trainable:
                self.assertGreater(result["gradient_norms"][name], 0)
                self.assertGreater(result["weight_change_norms"][name], 0)
            else:
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(parameter, before[name]), name)


if __name__ == "__main__":
    unittest.main()
