from app.storage.vector_db import search_embeddings
from app.storage.db import get_images_by_ids
from app.config import recall_limit, keyword_weight
import json

"""
Semantic search service.
"""

# Build text for rerank
def build_rerank_text(row: dict) -> str:
    return " ".join([
        row.get("caption", ""),
        " ".join(json.loads(row.get("objects", "[]"))),
        " ".join(json.loads(row.get("scene_tags", "[]"))),
    ])

# Keyword score
def keyword_score(query: str, text: str) -> float:
    query_terms = query.lower().split()
    text_lower = text.lower()

    if not query_terms:
        return 0.0

    matched = sum(1 for term in query_terms if term in text_lower)
    return matched / len(query_terms)


# Similarity (distance) score
def semantic_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)


# Search images by text query
def semantic_search(query: str, limit: int) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    # Step 1: recall candidates
    # Get most similar images with id and embedding distance info
    matches = search_embeddings(query, limit=recall_limit)
    if not matches:
        return []
    image_ids = [match["id"] for match in matches]

    # Get images records
    rows = get_images_by_ids(image_ids)

    # ID -> embedding distance info
    match_map = {match["id"]: match for match in matches}
    # ID -> image records
    row_map = {row["id"]: dict(row) for row in rows}

    scored_results = []

    # Step 2: rerank
    for image_id in image_ids:
        if image_id not in row_map:
            continue
        
        # Image records
        row = row_map[image_id]
        # Embedding distance info
        match = match_map[image_id]
        
        # Build text for rerank
        text = build_rerank_text(row)

        sem_score = semantic_score(match["distance"])
        key_score = keyword_score(query, text)

        # Final score = similarity + keyword
        final_score = sem_score + keyword_weight * key_score

        row["score"] = final_score
        scored_results.append(row)

    # Step 3: sort by final score
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results[:limit]