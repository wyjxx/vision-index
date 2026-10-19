"""Search the existing CLIP product index from the terminal."""

import argparse

from app.services.search import search_clip_products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Natural-language product query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    args = parser.parse_args(argv)
    query = args.query.strip()
    if not query:
        parser.error("query must not be empty")
    if args.limit <= 0:
        parser.error("limit must be positive")

    print(f"Searching: {query}", flush=True)
    results = search_clip_products(query, limit=args.limit)
    if not results:
        print("No results. Build the CLIP product index first.")
        return results

    for rank, product in enumerate(results, start=1):
        print(f"\n{rank}. {product['title']}")
        print(f"   Product ID: {product['product_id']}")
        print(f"   Similarity: {product['similarity']:.4f}")
        print(f"   Image: {product['image_path']}")
    return results


if __name__ == "__main__":
    main()
