"""Run one projection training step on a real query-image batch."""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from app.config import clip_model, model_cache_dir, product_catalog_path
from training.dataset import TrainingBatch, iter_training_batches, load_training_queries, read_jsonl
from training.loss import clip_contrastive_loss


def configure_projection_training(model: CLIPModel) -> dict[str, torch.nn.Parameter]:
    """Freeze CLIP except for the text and image projection weights."""
    model.requires_grad_(False)
    model.text_projection.requires_grad_(True)
    model.visual_projection.requires_grad_(True)
    # Disable dropout while keeping gradient tracking for trainable parameters.
    model.eval()
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_batch_inputs(batch: TrainingBatch, dataset_dir: Path, processor: CLIPProcessor):
    """Resolve product images and preprocess the batch texts and images."""
    catalog = {row["product_id"]: row["image_path"]
               for row in read_jsonl(dataset_dir / "products.jsonl")}
    images = []
    for product_id in batch.product_ids:
        with Image.open(dataset_dir / catalog[product_id]) as image:
            images.append(image.convert("RGB"))
    return processor(text=batch.texts, images=images, return_tensors="pt",
                     padding=True, truncation=True)


def compute_logits(model: CLIPModel, text_features, image_features):
    """Project features and compute temperature-scaled cosine similarities."""
    text_vectors = F.normalize(model.text_projection(text_features), dim=-1)
    image_vectors = F.normalize(model.visual_projection(image_features), dim=-1)
    return model.logit_scale.exp() * (text_vectors @ image_vectors.T)


def train_one_step(model: CLIPModel, inputs, positive_mask, optimizer) -> dict:
    """Update projection weights once and report loss, gradients, and weight changes."""
    # Find trainable parameters
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    # Save parameters copy before training
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    # Delete gradients from last batch
    optimizer.zero_grad(set_to_none=True)

    # Extract text + vision model features before projection
    with torch.no_grad():
        text_features = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output
        image_features = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output

    # Compute similarities and loss
    logits = compute_logits(model, text_features, image_features)
    loss = clip_contrastive_loss(logits, positive_mask)

    # Backpropagation and calculate gradient
    loss.backward()
    gradient_norms = {}
    for name, parameter in trainable.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or nonfinite gradient: {name}")
        gradient_norms[name] = parameter.grad.norm().item()
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise RuntimeError("Frozen parameters received gradients")

    # Update weights
    optimizer.step()

    # Compare losses using the same frozen features and batch.
    with torch.no_grad():
        loss_after = clip_contrastive_loss(compute_logits(model, text_features, image_features), positive_mask)
        changes = {name: (parameter - before[name]).norm().item() for name, parameter in trainable.items()}
    if not torch.isfinite(loss_after) or not all(math.isfinite(value) for value in changes.values()):
        raise RuntimeError("The update produced nonfinite values")

    return {"loss_before": loss.item(), "loss_after": loss_after.item(),
            "gradient_norms": gradient_norms, "weight_change_norms": changes,
            "frozen_parameters_with_grad": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=product_catalog_path.parent)
    parser.add_argument("--queries-per-batch", type=int, default=8)
    parser.add_argument("--images-per-query", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.queries_per_batch < 2:
        parser.error("queries-per-batch must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        queries = load_training_queries(args.dataset_dir)
        batch = next(iter_training_batches(queries, args.queries_per_batch, args.images_per_query, args.seed))
        if len(batch.query_ids) < 2:
            raise ValueError("Training requires at least two queries")
        print(f"Loading cached CLIP on {device}...", flush=True)
        processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=model_cache_dir, local_files_only=True)
        model = CLIPModel.from_pretrained(clip_model, cache_dir=model_cache_dir,
                                          local_files_only=True, dtype=torch.float32).to(device)
        trainable = configure_projection_training(model)
        inputs = load_batch_inputs(batch, args.dataset_dir, processor).to(device)
        positive_mask = torch.tensor(batch.positive_mask, dtype=torch.bool, device=device)
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.0)
        print(f"Batch: {len(batch.query_ids)} queries, {len(batch.product_ids)} images", flush=True)
        result = train_one_step(model, inputs, positive_mask, optimizer)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Training step failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print("Completed one step in memory; no checkpoint or index was saved.")
    return result


if __name__ == "__main__":
    main()
