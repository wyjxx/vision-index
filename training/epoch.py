"""Train CLIP projections for one full-pair epoch."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from app.config import clip_model, model_cache_dir, product_catalog_path
from training.dataset import TrainingQuery, iter_training_batches, load_training_queries
from training.step import configure_projection_training, load_batch_inputs, train_one_step


def train_one_epoch(
    model: CLIPModel,
    processor: CLIPProcessor,
    queries: list[TrainingQuery],
    dataset_dir: Path,
    optimizer,
    queries_per_batch: int = 64,
    images_per_query: int = 2,
    seed: int = 42,
) -> dict:
    """Update projections batch by batch and average the pre-update batch losses."""
    if queries_per_batch < 2:
        raise ValueError("queries_per_batch must be at least 2")
    device = next(model.parameters()).device
    batches = list(iter_training_batches(queries, queries_per_batch, images_per_query, seed))
    total_batches = len(batches)
    batch_losses = []
    trained_queries = set()
    trained_pairs = 0

    for batch_index, batch in enumerate(batches, start=1):
        inputs = load_batch_inputs(batch, dataset_dir, processor)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        positive_mask = torch.tensor(batch.positive_mask, dtype=torch.bool, device=device)
        # Reuse the optimizer so AdamW retains its state between batches.
        result = train_one_step(model, inputs, positive_mask, optimizer)
        batch_losses.append(result["loss_before"])
        trained_queries.update(batch.query_ids)
        trained_pairs += len(batch.pairs)
        print(f"Batch {batch_index}/{total_batches}: queries={len(batch.query_ids)}, "
              f"images={len(batch.product_ids)}, pairs={len(batch.pairs)}, "
              f"loss={result['loss_before']:.4f}", flush=True)

    if not batch_losses:
        raise ValueError("No batch had at least two queries")
    return {"batches": len(batch_losses), "trained_queries": len(trained_queries),
            "trained_pairs": trained_pairs,
            "mean_batch_loss": sum(batch_losses) / len(batch_losses)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=product_catalog_path.parent)
    parser.add_argument("--queries-per-batch", type=int, default=64)
    parser.add_argument("--images-per-query", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.queries_per_batch < 2 or args.images_per_query < 1:
        parser.error("queries-per-batch must be at least 2 and images-per-query must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        queries = load_training_queries(args.dataset_dir)
        print(f"Loading cached CLIP on {device}...", flush=True)
        processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=model_cache_dir, local_files_only=True)
        model = CLIPModel.from_pretrained(clip_model, cache_dir=model_cache_dir,
                                          local_files_only=True, dtype=torch.float32).to(device)
        trainable = configure_projection_training(model)
        # Initialize the optimizer once for the whole epoch.
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.0)
        result = train_one_epoch(model, processor, queries, args.dataset_dir, optimizer,
                                 args.queries_per_batch, args.images_per_query, args.seed)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Training epoch failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print("Completed one epoch in memory; no checkpoint or index was saved.")
    return result


if __name__ == "__main__":
    main()
