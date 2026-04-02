from app.storage.vector_db import search_embeddings
from app.storage.db import get_images_by_ids
from app.config import (
    recall_limit,
    global_weight,
    caption_weight,
    object_weight,
    scene_weight,
    attribute_weight,
)
from app.services.helper import parse_attributes, parse_json_list

"""
Semantic search service.
"""

# Return image records in fields: caption + objects + scene_tags + attributes
def build_search_fields(image_record: dict) -> dict[str, str]:
    """Extract searchable text per field for rerank."""
    attributes = parse_attributes(image_record.get("attributes"))

    return {
        "caption": image_record.get("caption", ""),
        "objects": " ".join(parse_json_list(image_record.get("objects"))),
        "scene_tags": " ".join(parse_json_list(image_record.get("scene_tags"))),
        "attributes": " ".join(attributes["lighting"] + attributes["color"]),
    }

# Keyword score
def keyword_score(query: str, text: str) -> float:
    query_terms = query.lower().split()
    text_lower = text.lower()

    if not query_terms:
        return 0.0

    matched = sum(1 for term in query_terms if term in text_lower)
    return matched / len(query_terms)


# Similarity score
def semantic_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)


# Calculate field keyword score
def field_keyword_scores(query: str, image_record: dict) -> dict[str, float]:
    """Calculate keyword score for each metadata field."""
    fields = build_search_fields(image_record)

    return {
        "caption": keyword_score(query, fields["caption"]),
        "objects": keyword_score(query, fields["objects"]),
        "scene_tags": keyword_score(query, fields["scene_tags"]),
        "attributes": keyword_score(query, fields["attributes"]),
    }


# Search images by text query
def semantic_search(query: str, limit: int) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    # Step 1: recall candidates
    # Get most similar images with id and embedding distance info
    vector_hits = search_embeddings(query, limit=recall_limit)
    if not vector_hits:
        return []
    image_ids = [vector_hit["id"] for vector_hit in vector_hits]

    # Get images records
    image_records = get_images_by_ids(image_ids)

    # ID -> embedding distance info
    vector_hit_by_id = {vector_hit["id"]: vector_hit for vector_hit in vector_hits}
    # ID -> image records
    image_record_by_id = {
        image_record["id"]: dict(image_record)
        for image_record in image_records
    }

    scored_results = []

    # Step 2: rerank
    for image_id in image_ids:
        if image_id not in image_record_by_id:
            continue
        
        # Image records
        image_record = image_record_by_id[image_id]
        # Embedding distance info
        vector_hit = vector_hit_by_id[image_id]

        global_score = semantic_score(vector_hit["distance"])
        
        field_scores = field_keyword_scores(query, image_record)

        final_score = (
            global_weight * global_score
            + caption_weight * field_scores["caption"]
            + object_weight * field_scores["objects"]
            + scene_weight * field_scores["scene_tags"]
            + attribute_weight * field_scores["attributes"]
        )

        image_record["final_score"] = final_score
        # For explainable result
        image_record["global_score"] = global_score
        image_record["field_score_sum"] = (
            caption_weight * field_scores["caption"]
            + object_weight * field_scores["objects"]
            + scene_weight * field_scores["scene_tags"]
            + attribute_weight * field_scores["attributes"]
        )
        image_record["field_score_each"] = field_scores
        scored_results.append(image_record)

    # Step 3: sort by final score
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    return scored_results[:limit]
