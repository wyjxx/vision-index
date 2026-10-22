"""Step 1: read training pairs and build a query-by-image positive mask.
Run python -m training.dataset to preview one batch without loading CLIP or images
The first experiment learns whether pairs match, without weighting by position.
"""

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingQuery:
    """One query and all its known positive products in the training split."""

    query_id: str
    text: str
    product_ids: list[str]


@dataclass
class TrainingBatch:
    """Rows are queries; columns are unique products; pairs track epoch coverage."""

    query_ids: list[str]
    texts: list[str]
    product_ids: list[str]
    positive_mask: list[list[bool]]
    pairs: list[tuple[str, str]]


def read_jsonl(path: Path):
    """Read rows without converting string IDs representing large integers to numbers."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def require_id(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def load_training_queries(dataset_dir: Path | str) -> list[TrainingQuery]:
    """Read queries.jsonl and train.jsonl"""
    dataset_dir = Path(dataset_dir)
    texts = {}
    for row in read_jsonl(dataset_dir / "queries.jsonl"):
        query_id = require_id(row["query_id"], "query_id")
        text = require_id(row["query"], "query text")
        if query_id in texts and texts[query_id] != text:
            raise ValueError(f"Conflicting query text: {query_id}")
        texts[query_id] = text

    groups = {}
    seen_pairs = set()
    for row in read_jsonl(dataset_dir / "train.jsonl"):
        query_id = require_id(row["query_id"], "query_id")
        product_id = require_id(row["product_id"], "product_id")
        if query_id not in texts:
            raise ValueError(f"Missing query text: {query_id}")
        pair = (query_id, product_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate training pair: {pair}")
        seen_pairs.add(pair)
        group = groups.setdefault(query_id, TrainingQuery(query_id, texts[query_id], []))
        group.product_ids.append(product_id)

    if not groups:
        raise ValueError("No training pairs found")
    return list(groups.values())


def build_training_batch(pairs: list[tuple[str, str]], queries: dict[str, TrainingQuery]) -> TrainingBatch:
    """Deduplicate queries and products, then mark all known training positives.

    Example mask: [[True, True, False], [False, True, True]].
    """
    query_ids = list(dict.fromkeys(query_id for query_id, _ in pairs))
    product_ids = list(dict.fromkeys(product_id for _, product_id in pairs))
    positive_mask = []
    for query_id in query_ids:
        known_positives = set(queries[query_id].product_ids)
        positive_mask.append([product_id in known_positives for product_id in product_ids])
    return TrainingBatch(query_ids, [queries[qid].text for qid in query_ids],
                         product_ids, positive_mask, pairs)


def iter_training_batches(
    queries: list[TrainingQuery],
    queries_per_batch: int = 64,
    images_per_query: int = 2,
    seed: int = 42,
):
    """Visit every pair once, rotating queries and merging single-query remainders."""
    for name, value in (("queries_per_batch", queries_per_batch), ("images_per_query", images_per_query)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not queries or any(not query.product_ids for query in queries):
        raise ValueError("Every training query must have positive products")
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Training query IDs must be unique")
    if queries_per_batch < 2 or len(queries) < 2:
        raise ValueError("Training batches require at least two queries")
    if any(len(set(query.product_ids)) != len(query.product_ids) for query in queries):
        raise ValueError("Training pairs must be unique")

    rng = random.Random(seed)
    ordered = list(queries)
    rng.shuffle(ordered)
    query_lookup = {query.query_id: query for query in ordered}
    remaining = {}
    for query in ordered:
        products = list(query.product_ids)
        rng.shuffle(products)
        remaining[query.query_id] = deque(products)

    # Return unfinished queries to the end of the queue.
    active = deque(query_lookup)
    batches = []
    while active:
        selected = [active.popleft() for _ in range(min(queries_per_batch, len(active)))]
        pairs = []
        for query_id in selected:
            products = remaining[query_id]
            pairs.extend((query_id, products.popleft())
                         for _ in range(min(images_per_query, len(products))))
            if products:
                active.append(query_id)

        # Merge a single-query remainder without dropping or repeating pairs.
        if len(selected) == 1:
            pairs = batches.pop().pairs + pairs
        batches.append(build_training_batch(pairs, query_lookup))

    yield from batches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "data" / "v2" / "marqo"
    parser.add_argument("--dataset-dir", type=Path, default=default_dir)
    parser.add_argument("--queries-per-batch", type=int, default=8)
    parser.add_argument("--images-per-query", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        queries = load_training_queries(args.dataset_dir)
        batches = iter_training_batches(queries, args.queries_per_batch, args.images_per_query, args.seed)
        batch = next(batches)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Training data inspection failed: {error}\n")
    print(f"Train: {len(queries)} queries, {sum(len(q.product_ids) for q in queries)} pairs")
    print("Batch preview (1 = known train positive; 0 = unlabeled, not confirmed negative):")
    print(json.dumps({
        "query_ids": batch.query_ids,
        "texts": batch.texts,
        "product_ids": batch.product_ids,
        "pairs": batch.pairs,
        "positive_mask": [[int(value) for value in row] for row in batch.positive_mask],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
