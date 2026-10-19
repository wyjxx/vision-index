"""Index all pending SQLite products using the same service as the web button."""

import argparse
from time import perf_counter

from filelock import Timeout

from app.config import clip_index_batch_size
from app.storage.sqlite_db import init_db


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=clip_index_batch_size,
                        help="Number of images encoded per batch")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")

    from app.services.index import index_unindexed_products

    try:
        init_db()
        started = perf_counter()
        processed = index_unindexed_products(batch_size=args.batch_size)
    except Timeout:
        parser.exit(1, "Product indexing is already running.\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Indexing failed: {exc}\n")

    print(f"Done: indexed {processed} pending products in {perf_counter() - started:.1f}s")
    return processed


if __name__ == "__main__":
    main()
