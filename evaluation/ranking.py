"""
Compute full exact cosine rankings
Example:
[
    {"product_id": "A", "similarity": 1.0, "model_rank": 1},
    {"product_id": "B", "similarity": 0.0, "model_rank": 2},
    {"product_id": "C", "similarity": -1.0, "model_rank": 3},
]

Main Function for Evaluation
Read data -> run model and compute ranking -> calculate metrics -> save results
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from app.config import clip_model, clip_chroma_collection, product_catalog_path
from evaluation.dataset import load_evaluation_queries
from evaluation.report import load_known_labels, write_query_outputs, build_summary


def rank_products(query_embedding, product_ids: list[str], product_embeddings) -> list[dict]:
    """Rank every candidate by cosine similarity"""
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
    return [
        {"product_id": product_ids[index], "similarity": float(similarities[index]), "model_rank": rank}
        for rank, index in enumerate(order, start=1)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=product_catalog_path.parent)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, help="Process only the first N queries")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for rankings and run config")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new --output-dir")

    from app.ai.clip import encode_texts
    from app.storage.clip_vector_db import get_clip_embeddings

    try:
        queries = load_evaluation_queries(args.dataset_dir, args.split)
        if args.limit is not None:
            queries = queries[:args.limit]
        known_labels = load_known_labels(args.dataset_dir, args.split)
        catalog_path = args.dataset_dir / "products.jsonl"
        with catalog_path.open(encoding="utf-8") as file:
            product_ids = [json.loads(line)["product_id"] for line in file if line.strip()]
        if not product_ids or len(set(product_ids)) != len(product_ids):
            raise ValueError("Catalog must contain unique product IDs")
        if any(not isinstance(pid, str) or not pid.strip() for pid in product_ids):
            raise ValueError("Catalog IDs must be nonempty strings")
        product_ids.sort()
        candidate_ids = set(product_ids)
        if any(not set(query["relevant_products"]).issubset(candidate_ids) for query in queries):
            raise ValueError("Evaluation labels contain products missing from the candidate catalog")
        images = np.asarray(get_clip_embeddings(product_ids), dtype=np.float32)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "model": clip_model, "collection": clip_chroma_collection,
            "dataset_dir": str(args.dataset_dir.resolve()), "split": args.split,
            "query_ids": [q["query_id"] for q in queries], "candidate_count": len(product_ids),
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "image_vectors_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
            "split_sha256": hashlib.sha256((args.dataset_dir / f"{args.split}.jsonl").read_bytes()).hexdigest(),
            "queries_sha256": hashlib.sha256((args.dataset_dir / "queries.jsonl").read_bytes()).hexdigest(),
            "retrieval": "exact cosine, full catalog, no filtering",
            "ranking_dtype": "float32", "tie_break": "product_id ascending (string)",
            "completed": False,
            "output_format": "compact-v2",
            "label_splits": ["train", "validation", "test"] if args.split == "test" else ["train", "validation"],
        }
        (args.output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Candidates: {len(product_ids)}, queries: {len(queries)}", flush=True)
        reports = []
        with (
            (args.output_dir / "pair_rankings.jsonl").open("x", encoding="utf-8") as pair_file,
            (args.output_dir / "per_query_metrics.jsonl").open("x", encoding="utf-8") as metric_file,
        ):
            for query in queries:
                embedding = encode_texts([query["query"]])[0]
                results = rank_products(embedding, product_ids, images)
                for result in results:
                    result["reference_position"] = query["relevant_products"].get(result["product_id"])
                reports.append(write_query_outputs(
                    query, results, known_labels[query["query_id"]], pair_file, metric_file,
                ))
                print(f"Ranked query {query['query_id']}: {len(results)} products", flush=True)
        summary = build_summary(reports, manifest)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest["completed"] = True
        (args.output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ranking failed: {error}\n")
    print(f"Saved compact rankings and full-ranking metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
