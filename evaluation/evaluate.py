"""Evaluate the current CLIP model against a full catalog."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from app.config import clip_model, model_cache_dir, product_catalog_path
from evaluation.dataset import load_evaluation_queries
from evaluation.report import build_summary, load_known_labels, write_query_outputs
from training.dataset import read_jsonl, require_id


def rank_products(query_embedding, product_ids: list[str], product_embeddings) -> list[dict]:
    """Rank every candidate by cosine similarity."""
    if not product_ids or any(not isinstance(pid, str) or not pid.strip() for pid in product_ids):
        raise ValueError("Candidate product IDs must be nonempty strings")
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Candidate product IDs must be unique")
    query = np.asarray(query_embedding, dtype=np.float32)
    images = np.asarray(product_embeddings, dtype=np.float32)
    if query.ndim != 1 or images.ndim != 2 or images.shape != (len(product_ids), query.size):
        raise ValueError("Expected one query vector and one equally sized vector per product")
    if not np.isfinite(query).all() or not np.isfinite(images).all():
        raise ValueError("Embeddings must contain only finite values")
    query_norm = np.linalg.norm(query)
    image_norms = np.linalg.norm(images, axis=1)
    if query_norm == 0 or np.any(image_norms == 0):
        raise ValueError("Embeddings must not have zero length")
    similarities = (images / image_norms[:, None]) @ (query / query_norm)
    order = np.lexsort((np.asarray(product_ids), -similarities))
    return [{"product_id": product_ids[index], "similarity": float(similarities[index]), "model_rank": rank}
            for rank, index in enumerate(order, start=1)]


@torch.inference_mode()
def encode_catalog_images(model, processor, dataset_dir: Path, batch_size: int):
    """Encode catalog images with the current model, generate image vectors"""
    products = list(read_jsonl(dataset_dir / "products.jsonl"))
    product_ids = [require_id(product["product_id"], "product_id") for product in products]
    if not products or len(set(product_ids)) != len(product_ids):
        raise ValueError("Catalog must contain unique product IDs")
    products.sort(key=lambda product: product["product_id"])
    device = next(model.parameters()).device
    vectors = []
    for start in range(0, len(products), batch_size):
        images = []
        for product in products[start:start + batch_size]:
            with Image.open(dataset_dir / product["image_path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=device, dtype=model.dtype)
        features = model.get_image_features(pixel_values=pixels).pooler_output
        vectors.append(F.normalize(features, dim=-1).float().cpu().numpy())
        if start % (batch_size * 32) == 0 or start + batch_size >= len(products):
            print(f"Catalog images: {min(start + batch_size, len(products))}/{len(products)}", flush=True)
    return [product["product_id"] for product in products], np.concatenate(vectors)


@torch.inference_mode()
def evaluate_model(model, processor, dataset_dir: Path, output_dir: Path,
                   model_name: str, batch_size: int = 128, split: str = "validation",
                   limit: int | None = None) -> dict:
    """Rank the full catalog and save reports for validation or test labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    queries = load_evaluation_queries(dataset_dir, split)
    if limit is not None:
        queries = queries[:limit]
    known_labels = load_known_labels(dataset_dir, split)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.eval()
    device = next(model.parameters()).device
    manifest = {
        "model": model_name, "split": split, "dataset_dir": str(dataset_dir.resolve()),
        "query_ids": [query["query_id"] for query in queries],
        "label_splits": ["train", "validation", "test"] if split == "test" else ["train", "validation"],
        "retrieval": "exact cosine, full catalog, no filtering",
        "ranking_dtype": "float32", "encoder_dtype": str(model.dtype),
        "tie_break": "product_id ascending (string)", "batch_size": batch_size,
        "output_format": "compact-v2", "completed": False,
    }
    files = {"catalog": "products", "queries": "queries", "split": split, "train": "train"}
    if split == "test":
        files["validation"] = "validation"
    for name, filename in files.items():
        manifest[f"{name}_sha256"] = hashlib.sha256((dataset_dir / f"{filename}.jsonl").read_bytes()).hexdigest()
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    product_ids, image_vectors = encode_catalog_images(model, processor, dataset_dir, batch_size)
    manifest["candidate_count"] = len(product_ids)
    manifest["image_vectors_sha256"] = hashlib.sha256(image_vectors.tobytes()).hexdigest()
    reports = []
    with (output_dir / "pair_rankings.jsonl").open("x", encoding="utf-8") as pair_file, \
         (output_dir / "per_query_metrics.jsonl").open("x", encoding="utf-8") as metric_file:
        for index, query in enumerate(queries, start=1):
            inputs = processor(text=[query["query"]], return_tensors="pt", padding=True, truncation=True)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            features = model.get_text_features(**inputs).pooler_output
            vector = F.normalize(features, dim=-1)[0].float().cpu().numpy()
            results = rank_products(vector, product_ids, image_vectors)
            reports.append(write_query_outputs(query, results, known_labels[query["query_id"]],
                                                pair_file, metric_file))
            if index % 25 == 0 or index == len(queries):
                print(f"{split.title()} queries: {index}/{len(queries)}", flush=True)

    summary = build_summary(reports, manifest)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest["completed"] = True
    config_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=product_catalog_path.parent)
    parser.add_argument("--model", "--checkpoint", default=clip_model, help="Model ID or local checkpoint directory")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--limit", type=int, help="Process only the first N queries")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new --output-dir")

    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = CLIPProcessor.from_pretrained(args.model, cache_dir=model_cache_dir, local_files_only=True)
        model = CLIPModel.from_pretrained(args.model, cache_dir=model_cache_dir, local_files_only=True,
                                          dtype=getattr(torch, args.dtype)).to(device)
        summary = evaluate_model(model, processor, args.dataset_dir, args.output_dir,
                                  args.model, args.batch_size, args.split, args.limit)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation to {args.output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    main()
