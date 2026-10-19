"""Coordinate CLIP query encoding and product retrieval."""

from app.ai.clip import encode_texts
from app.storage.clip_vector_db import search_clip_embeddings
from app.storage.sqlite_db import get_products_by_ids


def search_clip_products(query: str, limit: int = 10) -> list[dict]:
    """Encode a query, retrieve products, and attach cosine similarity."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    query = query.strip()
    if not query:
        return []
    embedding = encode_texts([query])[0]
    hits = search_clip_embeddings(embedding, limit)
    products = get_products_by_ids([hit["product_id"] for hit in hits])
    return [
        {**hit, **products[hit["product_id"]], "similarity": 1.0 - hit["distance"]}
        for hit in hits if hit["product_id"] in products
    ]
