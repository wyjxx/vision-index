"""Import Marqo data by sampling pairs and exporting query text and product images."""

import argparse
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory


def read_pairs(data_dir):
    """Yield string IDs and integer original positions, without reading images."""
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).rglob("in_domain-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No in_domain-*.parquet files under {data_dir}")

    for file in files:
        with pq.ParquetFile(file) as parquet:
            for batch in parquet.iter_batches(
                batch_size=4096, columns=["query_id", "product_id", "position"]
            ):
                for pair in batch.to_pylist():
                    for column in ("query_id", "product_id"):
                        value = pair[column]
                        if isinstance(value, bool) or not isinstance(value, (str, int)):
                            raise ValueError(f"{file.name}: {column} must be a string or integer")
                        if not str(value).strip():
                            raise ValueError(f"{file.name}: {column} must not be empty")
                        # Convert integers directly, preserving large product IDs.
                        pair[column] = str(value)
                    position = pair["position"]
                    if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= 100:
                        raise ValueError(f"{file.name}: position must be an integer from 1 to 100")
                    yield pair


def deduplicate_pairs(pairs):
    """Keep each pair once, rejecting duplicates with conflicting positions."""
    seen = {}
    unique_pairs = []
    duplicate_count = 0
    for pair in pairs:
        key = (pair["query_id"], pair["product_id"])
        if key in seen:
            if seen[key] != pair["position"]:
                raise ValueError(f"Conflicting positions for pair {key}")
            duplicate_count += 1
            continue
        seen[key] = pair["position"]
        unique_pairs.append(pair)
    return unique_pairs, duplicate_count


def sample_and_split_pairs(pairs, pair_count, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Sample unique pairs in random order, then slice into three partitions.

    Input pairs must already be deduplicated. Test receives the remainder after
    rounding train and validation counts down, so no sampled pairs are lost.
    """
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count <= 0:
        raise ValueError("pair-count must be a positive integer")
    if pair_count > len(pairs):
        raise ValueError(f"Requested {pair_count} pairs; available: {len(pairs)}")
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Train and validation ratios must be positive and sum to less than 1")
    train_count = int(pair_count * train_ratio)
    val_count = int(pair_count * val_ratio)
    if min(train_count, val_count, pair_count - train_count - val_count) == 0:
        raise ValueError("Too few sampled pairs for three nonempty partitions at these ratios")

    sampled = random.Random(seed).sample(pairs, pair_count)
    val_end = train_count + val_count
    return {
        "train": sampled[:train_count],
        "validation": sampled[train_count:val_end],
        "test": sampled[val_end:],
    }


def save_splits(splits, config, output_dir):
    """Save partitions without overwriting outputs; raw shards may share the directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "config.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Output already exists: {output_dir / name}")
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as file:
            for pair in rows:
                file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (output_dir / "config.json").open("x", encoding="utf-8") as file:
        file.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def export_queries(dataset_dir):
    """Create queries.jsonl without modifying the existing pair partitions."""
    import pyarrow.parquet as pq

    dataset_dir = Path(dataset_dir)
    output = dataset_dir / "queries.jsonl"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config = json.loads((dataset_dir / "config.json").read_text(encoding="utf-8"))
    selected_ids = set()
    for name in ("train", "validation", "test"):
        with (dataset_dir / f"{name}.jsonl").open(encoding="utf-8") as file:
            selected_ids.update(json.loads(line)["query_id"] for line in file)

    queries = {}
    for source in config["source_files"]:
        with pq.ParquetFile(source["path"]) as parquet:
            for batch in parquet.iter_batches(batch_size=4096, columns=["query_id", "query"]):
                for row in batch.to_pylist():
                    query_id = str(row["query_id"])
                    if query_id not in selected_ids:
                        continue
                    text = row["query"]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Missing query text for query_id={query_id}")
                    if query_id in queries and queries[query_id] != text:
                        raise ValueError(f"Conflicting text for query_id={query_id}")
                    queries[query_id] = text

    missing = selected_ids - queries.keys()
    if missing:
        raise ValueError(f"Query text not found for {len(missing)} IDs; examples: {sorted(missing)[:5]}")
    with output.open("x", encoding="utf-8") as file:
        for query_id in sorted(queries):
            file.write(json.dumps({"query_id": query_id, "query": queries[query_id]}, ensure_ascii=False) + "\n")
    return len(queries)


def image_extension(data):
    """Identify common source formats without re-encoding the image."""
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    raise ValueError("Unsupported image format")


def export_products(dataset_dir):
    """Write one image per selected product; keep existing partitions intact.

    If a product occurs repeatedly, use its first occurrence in source-file
    order. Temporary output is discarded if any selected image is missing.
    """
    import pyarrow.parquet as pq

    dataset_dir = Path(dataset_dir)
    for name in ("products.jsonl", "images"):
        if (dataset_dir / name).exists():
            raise FileExistsError(f"Output already exists: {dataset_dir / name}")
    config = json.loads((dataset_dir / "config.json").read_text(encoding="utf-8"))
    selected_ids = set()
    for name in ("train", "validation", "test"):
        with (dataset_dir / f"{name}.jsonl").open(encoding="utf-8") as file:
            selected_ids.update(json.loads(line)["product_id"] for line in file)
    if any(not isinstance(pid, str) or not pid.isascii() or not pid.isdigit() for pid in selected_ids):
        raise ValueError("Product IDs must be strings containing ASCII digits")

    products = {}
    total_bytes = 0
    with TemporaryDirectory(prefix=".products-", dir=dataset_dir) as temporary:
        staging = Path(temporary)
        (staging / "images").mkdir()
        for source in config["source_files"]:
            with pq.ParquetFile(source["path"]) as parquet:
                for batch in parquet.iter_batches(batch_size=128, columns=["product_id", "title", "image"]):
                    for row in batch.to_pylist():
                        product_id = str(row["product_id"])
                        if product_id not in selected_ids or product_id in products:
                            continue
                        data = (row["image"] or {}).get("bytes")
                        if not data:
                            raise ValueError(f"Missing embedded image for product_id={product_id}")
                        image_path = f"images/{product_id}{image_extension(data)}"
                        (staging / image_path).write_bytes(data)
                        products[product_id] = {
                            "product_id": product_id,
                            "title": row["title"],
                            "image_path": image_path,
                        }
                        total_bytes += len(data)
        missing = selected_ids - products.keys()
        if missing:
            raise ValueError(f"Products not found for {len(missing)} IDs; examples: {sorted(missing)[:5]}")
        with (staging / "products.jsonl").open("x", encoding="utf-8") as file:
            for product_id in sorted(products):
                file.write(json.dumps(products[product_id], ensure_ascii=False) + "\n")
        (staging / "images").rename(dataset_dir / "images")
        (staging / "products.jsonl").rename(dataset_dir / "products.jsonl")
    return len(products), total_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing in_domain Parquet shards")
    parser.add_argument("--pair-count", type=int, required=True, help="Number of unique pairs to sample")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/v2/marqo"))
    args = parser.parse_args(argv)
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("Output path must be a directory")
    # The raw shard may share this directory, but generated data must never be overwritten.
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "config.json",
                 "queries.jsonl", "products.jsonl", "images"):
        if (args.output_dir / name).exists():
            parser.error(f"Output already exists: {args.output_dir / name}; choose a new --output-dir")

    try:
        print(f"Reading ID pairs from {args.data_dir} ...")
        pairs, duplicate_count = deduplicate_pairs(read_pairs(args.data_dir))
        splits = sample_and_split_pairs(
            pairs, args.pair_count, args.train_ratio, args.val_ratio, args.seed,
        )
        config = {
            "data_dir": str(args.data_dir.resolve()),
            "source_files": [
                {"path": str(file.resolve()), "size_bytes": file.stat().st_size,
                 "mtime_ns": file.stat().st_mtime_ns}
                for file in sorted(args.data_dir.rglob("in_domain-*.parquet"))
            ],
            "pair_count": args.pair_count,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(1 - args.train_ratio - args.val_ratio, 12),
            "seed": args.seed,
            "deduplication_key": ["query_id", "product_id"],
            "pair_fields": ["query_id", "product_id", "position"],
            "raw_pair_count": len(pairs) + duplicate_count,
            "unique_pair_count": len(pairs),
            "duplicates_removed": duplicate_count,
            "split_counts": {name: len(rows) for name, rows in splits.items()},
        }
        save_splits(splits, config, args.output_dir)
        print("Exporting query text ...")
        query_count = export_queries(args.output_dir)
        print("Exporting product titles and original images ...")
        product_count, image_bytes = export_products(args.output_dir)
    except ImportError:
        parser.error("Install project dependencies: python -m pip install -r requirements.txt")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    print(f"Raw pairs: {len(pairs) + duplicate_count}")
    print(f"Unique pairs: {len(pairs)}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Seed: {args.seed}, sampled pairs: {args.pair_count}")
    for name, rows in splits.items():
        queries = len({pair["query_id"] for pair in rows})
        products = len({pair["product_id"] for pair in rows})
        print(f"{name}: pairs={len(rows)}, queries={queries}, products={products}")
    print(f"Exported queries: {query_count}, products: {product_count}")
    print(f"Original images: {image_bytes / 1_000_000:.1f} MB")
    print(f"Saved complete sample to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
