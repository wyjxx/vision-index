"""Read and build data for evaluation"""

import argparse
import json
from pathlib import Path

from app.config import product_catalog_path


def load_evaluation_queries(dataset_dir: Path | str, split: str = "validation") -> list[dict]:
    """Read queries.jsonl and test or validation.jsonl
    Build:
    {
    "query_id": "q1",
    "query": "orange juicer",
    "relevant_products": {
        "A": 1,
        "C": 51,
    },}
    """
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    dataset_dir = Path(dataset_dir)
    query_texts = {}
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id, text = row["query_id"], row["query"]
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError("query_id must be a nonempty string")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing query text for {query_id}")
            if query_id in query_texts and query_texts[query_id] != text:
                raise ValueError(f"Conflicting query text for {query_id}")
            query_texts[query_id] = text

    groups = {}
    with (dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id, product_id, position = row["query_id"], row["product_id"], row["position"]
            if any(not isinstance(value, str) or not value.strip() for value in (query_id, product_id)):
                raise ValueError("Query and product IDs must be nonempty strings")
            if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= 100:
                raise ValueError("position must be an integer from 1 to 100")
            if query_id not in query_texts:
                raise ValueError(f"Query text not found for {query_id}")
            group = groups.setdefault(query_id, {
                "query_id": query_id,
                "query": query_texts[query_id],
                "relevant_products": {},
            })
            if product_id in group["relevant_products"]:
                raise ValueError(f"Duplicate pair: {query_id}, {product_id}")
            group["relevant_products"][product_id] = position
    if not groups:
        raise ValueError(f"No evaluation pairs found in {split}.jsonl")

    return list(groups.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=product_catalog_path.parent)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(argv)
    try:
        queries = load_evaluation_queries(args.dataset_dir, args.split)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    counts = [len(query["relevant_products"]) for query in queries]
    print(f"Split: {args.split}")
    print(f"Queries: {len(queries)}, pairs: {sum(counts)}")
    print(f"Relevant products per query: min={min(counts)}, max={max(counts)}")
    print("Data inspection only; no model calls or retrieval metrics.")


if __name__ == "__main__":
    main()
