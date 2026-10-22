"""Multi-positive contrastive loss for query-image batches."""

import torch
from torch import Tensor
from torch.nn import functional as F


def multi_positive_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    """Calculate Unidirectional loss."""
    if logits.ndim != 2 or 0 in logits.shape or logits.shape != positive_mask.shape:
        raise ValueError("Expected nonempty logits and positive_mask with the same 2D shape")
    if positive_mask.dtype != torch.bool or positive_mask.device != logits.device:
        raise ValueError("positive_mask must be boolean and on the same device as logits")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must contain finite floating-point values")
    positive_counts = positive_mask.sum(dim=1)
    if (positive_counts == 0).any():
        raise ValueError("Every row must contain at least one positive")

    log_probs = F.log_softmax(logits.float(), dim=1)
    positive_log_probs = log_probs.masked_fill(~positive_mask, 0.0)
    row_losses = -positive_log_probs.sum(dim=1) / positive_counts
    return row_losses.mean()


def clip_contrastive_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    """
    Calculate bidirectional contrastive loss.
    Average text-to-image and image-to-text losses for a query-by-image matrix."""
    text_loss = multi_positive_loss(logits, positive_mask)
    image_loss = multi_positive_loss(logits.T, positive_mask.T)
    return (text_loss + image_loss) / 2
