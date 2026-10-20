"""Output report and summary"""

import json
from pathlib import Path
from statistics import mean, median

from evaluation.metrics import recall_at_k, ndcg_at_k


def load_known_labels(dataset_dir: Path, split: str) -> dict[str, dict[str, int]]:
    """Include all available split labels for test; never read test for validation."""
    splits = ("train", "validation", "test") if split == "test" else ("train", "validation")
    labels = {}
    for name in splits:
        with (dataset_dir / f"{name}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                products = labels.setdefault(row["query_id"], {})
                if row["product_id"] in products:
                    raise ValueError(f"Duplicate pair across splits: {row['query_id']}, {row['product_id']}")
                products[row["product_id"]] = row["position"]
    return labels


def build_query_report(query: dict, results: list[dict], known_labels: dict[str, int]) -> dict:
    """Compute Recall and ndcg for each query"""
    ids = [r["product_id"] for r in results]
    if [r["model_rank"] for r in results] != list(range(1, len(results) + 1)):
        raise ValueError("Expected a complete ranking with consecutive original ranks")
    if not set(query["relevant_products"]).issubset(ids) or not set(known_labels).issubset(ids):
        raise ValueError("Labeled products are missing from the ranking")
    item = {"query_id": query["query_id"], "query": query["query"],
            "label_count": len(query["relevant_products"]), "all_label_count": len(known_labels)}
    for scope, labels in (("test_pairs", query["relevant_products"]), ("all_known_pairs", known_labels)):
        item[scope] = {
            **{f"recall@{k}": recall_at_k(ids, labels, k) for k in (10, 50, 100)},
            **{f"ndcg@{k}": ndcg_at_k(ids, labels, k) for k in dict.fromkeys((10, 50, 100, len(ids)))},
        }
    ranks = [r["model_rank"] for r in results if r["product_id"] in query["relevant_products"]]
    item["mean_labeled_pair_rank"] = mean(ranks)
    item["labeled_pair_ranks"] = ranks
    return item


def write_query_outputs(query, results, known_labels, pair_file, metric_file):
    """Save and write result for each query."""
    report = build_query_report(query, results, known_labels)
    for result in results:
        position = query["relevant_products"].get(result["product_id"])
        if position is not None:
            row = {"query_id": query["query_id"], "query": query["query"],
                   **result, "reference_position": position}
            pair_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metric_file.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def build_summary(reports: list[dict], manifest: dict) -> dict:
    '''Summary all queries results'''
    summary = {
        "model": manifest["model"], "split": manifest["split"], "queries": len(reports),
        "candidates": manifest["candidate_count"],
        "protocol": "exact full-catalog ranking; no filtering; linear gain=101-position; macro query average",
    }
    for scope in ("test_pairs", "all_known_pairs"):
        summary[scope] = {key: mean(r[scope][key] for r in reports) for key in reports[0][scope]}
    summary["test_pairs"]["pair_count"] = sum(r["label_count"] for r in reports)
    summary["rank_diagnostics"] = {
        "query_macro_mean_rank": mean(r["mean_labeled_pair_rank"] for r in reports),
        "pair_median_rank": median(rank for r in reports for rank in r["labeled_pair_ranks"]),
    }
    summary["scope_note"] = "test_pairs means the selected split's labels; all_known includes training labels."
    summary["latency"] = "Not measured; JSON writing and full sorting are not serving latency."
    return summary
