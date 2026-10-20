"""Store and retrieve precomputed CLIP vectors without calling models."""

from functools import lru_cache

import chromadb

from app.config import clip_chroma_collection, clip_chroma_dir, clip_model


@lru_cache(maxsize=1)
def get_clip_collection():
    """Open persistent cosine storage dedicated to this CLIP model."""
    client = chromadb.PersistentClient(path=str(clip_chroma_dir))
    collection = client.get_or_create_collection(
        name=clip_chroma_collection,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"clip_model": clip_model},
    )
    if (collection.metadata or {}).get("clip_model") != clip_model:
        raise ValueError("Collection model differs from CLIP_MODEL; use a new CLIP_COLLECTION")
    return collection


def upsert_clip_embeddings(products: list[dict], embeddings: list[list[float]]) -> int:
    """Store one batch of product records and their already computed vectors."""
    if len(products) != len(embeddings):
        raise ValueError("Each product must have exactly one embedding")
    if not products:
        return 0
    ids = [product["product_id"] for product in products]
    if any(not isinstance(pid, str) or not pid.strip() for pid in ids):
        raise ValueError("product_id must be a nonempty string")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate product IDs within the batch")

    collection = get_clip_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=[
            {"title": product.get("title") or "", "image_path": product["image_path"]}
            for product in products
        ],
    )
    return len(products)


def search_clip_embeddings(embedding: list[float], limit: int = 10) -> list[dict]:
    """Return products ordered by cosine distance (smaller is better)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    collection = get_clip_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(limit, count),
        include=["metadatas", "distances"],
    )
    return [
        {"product_id": product_id, **metadata, "distance": distance}
        for product_id, metadata, distance in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0],
        )
    ]


def get_clip_embeddings(product_ids: list[str]) -> list[list[float]]:
    """Read existing image vectors in the requested ID order, without encoding images."""
    if not product_ids:
        return []
    # Evaluation must fail on a missing index instead of creating an empty one.
    if not (clip_chroma_dir / "chroma.sqlite3").is_file():
        raise ValueError("CLIP index does not exist; index the catalog first")
    client = chromadb.PersistentClient(path=str(clip_chroma_dir))
    collection = client.get_collection(clip_chroma_collection, embedding_function=None)
    if (collection.metadata or {}).get("clip_model") != clip_model:
        raise ValueError("Collection model differs from CLIP_MODEL")
    vectors = {}
    for start in range(0, len(product_ids), 500):
        batch = collection.get(ids=product_ids[start:start + 500], include=["embeddings"])
        vectors.update(zip(batch["ids"], batch["embeddings"]))
    missing = set(product_ids) - vectors.keys()
    if missing:
        raise ValueError(f"Missing CLIP vectors for {len(missing)} catalog products")
    return [vectors[product_id] for product_id in product_ids]
