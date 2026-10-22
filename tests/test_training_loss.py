"""Check contrastive loss values and gradients with small matrices."""

import math
import unittest

import torch
from torch.nn import functional as F

from training.loss import clip_contrastive_loss, multi_positive_loss


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        # q1 matches A and B; q2 matches C.
        self.mask = torch.tensor([[True, True, False], [False, False, True]])

    def test_uniform_scores_match_hand_calculation(self):
        logits = torch.zeros(2, 3)
        # Uniform text-to-image probability is 1/3; image-to-text is 1/2.
        self.assertAlmostEqual(multi_positive_loss(logits, self.mask).item(), math.log(3), places=6)
        expected = (math.log(3) + math.log(2)) / 2
        self.assertAlmostEqual(clip_contrastive_loss(logits, self.mask).item(), expected, places=6)

    def test_better_matching_scores_reduce_loss(self):
        good = torch.tensor([[2.0, 2.0, -2.0], [-2.0, -2.0, 2.0]])
        self.assertLess(clip_contrastive_loss(good, self.mask),
                        clip_contrastive_loss(torch.zeros(2, 3), self.mask))

    def test_gradient_step_lowers_loss_and_supports_both_positives(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        loss = clip_contrastive_loss(logits, self.mask)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # Starting from equal scores, gradient descent raises every positive score.
        self.assertTrue((logits.grad[self.mask] < 0).all())
        self.assertTrue((logits.grad[~self.mask] > 0).all())
        with torch.no_grad():
            updated = logits - 0.1 * logits.grad
        self.assertLess(clip_contrastive_loss(updated, self.mask).item(), loss.item())

    def test_one_positive_per_row_matches_cross_entropy(self):
        logits = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        mask = torch.eye(2, dtype=torch.bool)
        labels = torch.arange(2)
        expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        torch.testing.assert_close(clip_contrastive_loss(logits, mask), expected)

    def test_invalid_masks_and_nonfinite_scores_are_rejected(self):
        cases = [
            (torch.zeros(2, 3), self.mask[:, :2]),
            (torch.zeros(2, 3), self.mask.float()),
            (torch.zeros(2, 3), torch.tensor([[False, False, False], [True, True, True]])),
            # Each image also needs a positive query for the reverse direction.
            (torch.zeros(2, 3), torch.tensor([[True, False, False], [False, True, False]])),
            (torch.full((2, 3), float("nan")), self.mask),
            (torch.empty(0, 3), torch.empty(0, 3, dtype=torch.bool)),
        ]
        for logits, mask in cases:
            with self.subTest(shape=logits.shape, mask=mask), self.assertRaises(ValueError):
                clip_contrastive_loss(logits, mask)


if __name__ == "__main__":
    unittest.main()
