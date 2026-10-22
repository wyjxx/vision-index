"""Train CLIP projections across epochs and select checkpoints using validation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import CLIPModel, CLIPProcessor

from app.config import clip_model, model_cache_dir, product_catalog_path
from training.checkpoint import load_training_state, save_training_state
from training.dataset import load_training_queries
from training.epoch import train_one_epoch
from training.step import configure_projection_training
from evaluation.evaluate import evaluate_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Checkpoint to continue; epochs is the total target")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--queries-per-batch", type=int)
    parser.add_argument("--images-per-query", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args(argv)
    state = None
    if args.resume:
        try:
            state = load_training_state(args.resume)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            parser.error(str(error))
        if args.epochs <= state["epoch"]:
            parser.error(f"epochs must exceed the completed epoch {state['epoch']}")

    # Inherit saved settings when resuming; otherwise use training defaults.
    defaults = {"queries_per_batch": 64, "images_per_query": 2,
                "eval_batch_size": 128, "seed": 42, "learning_rate": 1e-4}
    for name, default in defaults.items():
        saved_value = state["run"][name] if state else default
        value = getattr(args, name)
        if state and value is not None and value != saved_value:
            parser.error(f"Resume requires the original {name}={saved_value}")
        setattr(args, name, saved_value if value is None else value)
    if args.dataset_dir is None:
        args.dataset_dir = Path(state["run"]["dataset_dir"]) if state else product_catalog_path.parent
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.queries_per_batch < 2 or args.images_per_query < 1 or args.eval_batch_size < 1:
        parser.error("Invalid training or evaluation batch size")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new --output-dir")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"base_model": clip_model, "dataset_dir": str(args.dataset_dir.resolve()),
              "seed": args.seed, "learning_rate": args.learning_rate,
              "queries_per_batch": args.queries_per_batch, "images_per_query": args.images_per_query,
              "eval_batch_size": args.eval_batch_size, "epochs": args.epochs,
              "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
              "device": str(device), "dtype": "float32", "weight_decay": 0.0,
              "selection_metric": "validation.test_pairs.ndcg@10", "completed": False}
    try:
        dataset_hashes = {name: hashlib.sha256((args.dataset_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                          for name in ("train", "validation", "queries", "products")}
        if state and dataset_hashes != state["run"]["dataset_sha256"]:
            raise ValueError("Dataset files differ from the saved training run")
        queries = load_training_queries(args.dataset_dir)
        if state:
            result = state["run"]
            result.update({"epochs": args.epochs, "completed": False,
                           "dataset_dir": str(args.dataset_dir.resolve()),
                           "resumed_from": str(args.resume.resolve()), "device": str(device),
                           "torch_version": str(torch.__version__), "transformers_version": transformers.__version__})
        result["dataset_sha256"] = dataset_hashes
        args.output_dir.mkdir(parents=True, exist_ok=False)
        run_path = args.output_dir / "run.json"
        run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        model_source = args.resume if state else clip_model
        print(f"Loading CLIP from {model_source} on {device}...", flush=True)
        processor = CLIPProcessor.from_pretrained(model_source, cache_dir=model_cache_dir, local_files_only=True)
        model = CLIPModel.from_pretrained(model_source, cache_dir=model_cache_dir,
                                          local_files_only=True, dtype=torch.float32).to(device)
        trainable = configure_projection_training(model)
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=0.0)

        if state:
            optimizer.load_state_dict(state["optimizer"])
            # Restore RNG states after model initialization.
            torch.set_rng_state(state["torch_rng_state"])
            if torch.cuda.is_available() and state["cuda_rng_states"]:
                torch.cuda.set_rng_state_all(state["cuda_rng_states"])
            baseline = result["validation_before"]
            start_epoch = state["epoch"] + 1
        else:
            print("Evaluating pretrained validation baseline...", flush=True)
            baseline = evaluate_model(model, processor, args.dataset_dir,
                                       args.output_dir / "validation_before", clip_model, args.eval_batch_size,
                                       split="validation")
            result["validation_before"] = baseline
            result["history"] = []
            result["best_epoch"] = 0
            result["best_model"] = clip_model
            result["best_score"] = baseline["test_pairs"]["ndcg@10"]
            start_epoch = 1
        before_score = baseline["test_pairs"]["ndcg@10"]
        run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        # Continue with the same model and optimizer across epochs.
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_seed = args.seed + epoch - 1
            print(f"Epoch {epoch}/{args.epochs}: sampler seed={epoch_seed}", flush=True)
            training = train_one_epoch(model, processor, queries, args.dataset_dir, optimizer,
                                        args.queries_per_batch, args.images_per_query, epoch_seed)

            # Save each epoch before validation without overwriting earlier checkpoints.
            checkpoint_dir = args.output_dir / f"checkpoint_epoch_{epoch}"
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint: {checkpoint_dir}", flush=True)
            validation = evaluate_model(model, processor, args.dataset_dir,
                                         args.output_dir / f"validation_epoch_{epoch}",
                                         str(checkpoint_dir.resolve()), args.eval_batch_size, split="validation")
            score = validation["test_pairs"]["ndcg@10"]
            deltas = {key: value - baseline["test_pairs"][key]
                      for key, value in validation["test_pairs"].items() if key != "pair_count"}
            result["history"].append({"epoch": epoch, "seed": epoch_seed, "training": training,
                                       "validation": validation, "held_out_metric_deltas": deltas})
            # Keep the earlier model when validation scores tie.
            if score > result["best_score"]:
                result["best_score"] = score
                result["best_epoch"] = epoch
                result["best_model"] = str(checkpoint_dir.resolve())
            result["completed"] = epoch == args.epochs
            save_training_state(checkpoint_dir, optimizer, epoch, result)
            run_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Epoch {epoch}: validation held-out nDCG@10={score:.4f}; "
                  f"best epoch={result['best_epoch']}", flush=True)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Training run failed: {error}\n")

    print(f"Validation held-out nDCG@10: baseline={before_score:.4f}; best={result['best_score']:.4f}")
    print(f"Best epoch: {result['best_epoch']}; model: {result['best_model']}")
    print(f"Reports saved to {args.output_dir.resolve()}")
    return result


if __name__ == "__main__":
    main()
