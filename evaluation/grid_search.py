import json
from itertools import product
from pathlib import Path

from evaluation.eval_search import run_full_evaluation


# Paths
EVAL_DIR = Path("evaluation")
GRID_RESULT_PATH = EVAL_DIR / "grid_search_result.json"

# Grid search space
GLOBAL_WEIGHT_CANDIDATES = [0.5, 1.0, 1.5]
FIELD_WEIGHT_CANDIDATES = [0.05, 0.1, 0.2]
TOP_N_TO_PRINT = 10


def generate_weight_candidates() -> list[dict[str, float]]:
    """Build the requested 3 * 3 * 3 * 3 * 3 grid."""
    candidates = []

    for values in product(
        GLOBAL_WEIGHT_CANDIDATES,
        FIELD_WEIGHT_CANDIDATES,
        FIELD_WEIGHT_CANDIDATES,
        FIELD_WEIGHT_CANDIDATES,
        FIELD_WEIGHT_CANDIDATES,
    ):
        (
            global_weight,
            caption_weight,
            object_weight,
            scene_weight,
            attribute_weight,
        ) = values

        candidates.append(
            {
                "global_weight": global_weight,
                "caption_weight": caption_weight,
                "object_weight": object_weight,
                "scene_weight": scene_weight,
                "attribute_weight": attribute_weight,
            }
        )

    return candidates


def rank_key(item: dict) -> tuple[float, float, float]:
    """Sort by precision first, then recall, then top1."""
    summary = item["summary"]
    return (
        summary["avg_precision_at_5"],
        summary["avg_recall_at_5"],
        summary["avg_top1_accuracy"],
    )


def save_grid_results(
    path: Path,
    candidates: list[dict],
    ranked_results: list[dict],
) -> None:
    """Save the full grid search result to json."""
    payload = {
        # Record the evaluated search space.
        "search_space": {
            "global_weight": GLOBAL_WEIGHT_CANDIDATES,
            "caption_weight": FIELD_WEIGHT_CANDIDATES,
            "object_weight": FIELD_WEIGHT_CANDIDATES,
            "scene_weight": FIELD_WEIGHT_CANDIDATES,
            "attribute_weight": FIELD_WEIGHT_CANDIDATES,
        },
        "candidate_count": len(candidates),
        # Ranking priority requested by the user.
        "ranking_priority": [
            "avg_precision_at_5",
            "avg_recall_at_5",
            "avg_top1_accuracy",
        ],
        # Save both best and full ranked results.
        "best_result": ranked_results[0] if ranked_results else None,
        "top_results": ranked_results[:TOP_N_TO_PRINT],
        "all_results": ranked_results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_top_results(ranked_results: list[dict]) -> None:
    """Print the best few parameter combinations."""
    print("\n" + "=" * 60)
    print("Grid Search Summary")
    print("=" * 60)

    if not ranked_results:
        print("No grid search results produced.")
        return

    best = ranked_results[0]
    print("Best Weights:")
    print(best["weights"])
    print(
        "Best Metrics: "
        f"Precision@5={best['summary']['avg_precision_at_5']:.4f}, "
        f"Recall@5={best['summary']['avg_recall_at_5']:.4f}, "
        f"Top1={best['summary']['avg_top1_accuracy']:.4f}"
    )

    print("\nTop Candidates:")
    for index, item in enumerate(ranked_results[:TOP_N_TO_PRINT], start=1):
        summary = item["summary"]
        print(
            f"{index}. weights={item['weights']} | "
            f"precision@5={summary['avg_precision_at_5']:.4f} | "
            f"recall@5={summary['avg_recall_at_5']:.4f} | "
            f"top1={summary['avg_top1_accuracy']:.4f}"
        )


def main() -> None:
    # Build all parameter combinations.
    candidates = generate_weight_candidates()
    results = []

    # Run one full evaluation for each weight set.
    for index, weights in enumerate(candidates, start=1):
        _, summary = run_full_evaluation(weights=weights)
        results.append(
            {
                "rank_key": {
                    "avg_precision_at_5": summary["avg_precision_at_5"],
                    "avg_recall_at_5": summary["avg_recall_at_5"],
                    "avg_top1_accuracy": summary["avg_top1_accuracy"],
                },
                "weights": weights,
                "summary": summary,
            }
        )
        print(
            f"[{index}/{len(candidates)}] "
            f"weights={weights} "
            f"precision@5={summary['avg_precision_at_5']:.4f} "
            f"recall@5={summary['avg_recall_at_5']:.4f} "
            f"top1={summary['avg_top1_accuracy']:.4f}"
        )

    # Rank all candidates by the requested metric priority.
    ranked_results = sorted(results, key=rank_key, reverse=True)

    # Save and print final grid search summary.
    save_grid_results(GRID_RESULT_PATH, candidates, ranked_results)
    print_top_results(ranked_results)
    print(f"\nSaved grid search results to: {GRID_RESULT_PATH}")


if __name__ == "__main__":
    main()
