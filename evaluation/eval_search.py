# evaluation/eval_search.py

import json
from pathlib import Path

from app.services.search import semantic_search


# Paths
EVAL_DIR = Path("evaluation")
GOLDEN_PATH = EVAL_DIR / "golden_queries.json"
RESULT_PATH = EVAL_DIR / "result.json"

# Search config
SEARCH_LIMIT = 10
TOP_K = 5


def load_golden_queries(path: Path) -> list[dict]:
    """Load golden query annotations from json file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_best_files(item: dict) -> list[str]:
    """
    Get best files from one query item.

    Support both:
    - best_files: [...]
    - best_file: [...]
    - best_file: "001.jpg"
    """
    if "best_files" in item:
        best = item["best_files"]
    else:
        best = item.get("best_file", [])

    if isinstance(best, str):
        return [best]

    if isinstance(best, list):
        return best

    return []


def extract_file_names(results: list[dict]) -> list[str]:
    """Extract file_name list from search results."""
    file_names = []

    for row in results:
        file_name = row.get("file_name")
        if file_name:
            file_names.append(file_name)

    return file_names


def calc_top1_accuracy(predicted_files: list[str], best_files: list[str]) -> float:
    """Return 1.0 if top1 is in best_files, else 0.0."""
    if not predicted_files or not best_files:
        return 0.0

    return 1.0 if predicted_files[0] in best_files else 0.0


def calc_precision_at_k(
    predicted_files: list[str],
    relevant_files: list[str],
    k: int,
) -> float:
    """Calculate precision@k."""
    top_k_files = predicted_files[:k]
    if not top_k_files:
        return 0.0

    hits = sum(1 for file_name in top_k_files if file_name in relevant_files)
    return hits / k


def calc_recall_at_k(
    predicted_files: list[str],
    relevant_files: list[str],
    k: int,
) -> float:
    """Calculate recall@k."""
    if not relevant_files:
        return 0.0

    top_k_files = predicted_files[:k]
    hits = sum(1 for file_name in top_k_files if file_name in relevant_files)
    return hits / len(relevant_files)


def evaluate_one_query(item: dict) -> dict:
    """Run search and evaluate one query."""
    query = item["query"]
    query_type = item.get("type", "")
    relevant_files = item.get("relevant_files", [])
    best_files = get_best_files(item)

    # Run search
    results = semantic_search(query, limit=SEARCH_LIMIT)

    # Extract ordered file names
    predicted_files = extract_file_names(results)

    # Calculate metrics
    top1_acc = calc_top1_accuracy(predicted_files, best_files)
    precision_at_5 = calc_precision_at_k(predicted_files, relevant_files, TOP_K)
    recall_at_5 = calc_recall_at_k(predicted_files, relevant_files, TOP_K)

    # Build result record
    return {
        "query": query,
        "type": query_type,
        "relevant_files": relevant_files,
        "best_files": best_files,
        "predicted_files": predicted_files,
        "top1_accuracy": top1_acc,
        "precision_at_5": round(precision_at_5, 4),
        "recall_at_5": round(recall_at_5, 4),
    }


def calc_average(items: list[dict], key: str) -> float:
    """Calculate average value for one metric key."""
    if not items:
        return 0.0

    return sum(item[key] for item in items) / len(items)


def save_results(path: Path, results: list[dict], summary: dict) -> None:
    """Save detailed results and summary to json file."""
    data = {
        "summary": summary,
        "queries": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_query_result(item: dict) -> None:
    """Print one query result to console."""
    print(f"\nQuery: {item['query']}")
    print(f"Type: {item['type']}")
    print(f"Predicted: {item['predicted_files'][:TOP_K]}")
    print(f"Relevant: {item['relevant_files']}")
    print(f"Best: {item['best_files']}")
    print(f"Top1 Accuracy: {item['top1_accuracy']:.2f}")
    print(f"Precision@5: {item['precision_at_5']:.2f}")
    print(f"Recall@5: {item['recall_at_5']:.2f}")


def main() -> None:
    """Run full search evaluation."""
    golden_queries = load_golden_queries(GOLDEN_PATH)

    all_results = []

    # Evaluate each query
    for item in golden_queries:
        result = evaluate_one_query(item)
        all_results.append(result)
        print_query_result(result)

    # Calculate summary metrics
    summary = {
        "query_count": len(all_results),
        "search_limit": SEARCH_LIMIT,
        "metric_top_k": TOP_K,
        "avg_top1_accuracy": round(calc_average(all_results, "top1_accuracy"), 4),
        "avg_precision_at_5": round(calc_average(all_results, "precision_at_5"), 4),
        "avg_recall_at_5": round(calc_average(all_results, "recall_at_5"), 4),
    }

    # Save full results
    save_results(RESULT_PATH, all_results, summary)

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Query Count: {summary['query_count']}")
    print(f"Search Limit: {summary['search_limit']}")
    print(f"Metric Top K: {summary['metric_top_k']}")
    print(f"Average Top1 Accuracy: {summary['avg_top1_accuracy']:.4f}")
    print(f"Average Precision@5:   {summary['avg_precision_at_5']:.4f}")
    print(f"Average Recall@5:      {summary['avg_recall_at_5']:.4f}")
    print(f"\nSaved result to: {RESULT_PATH}")


if __name__ == "__main__":
    main()