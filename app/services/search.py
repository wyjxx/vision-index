from app.storage.vector_db import search_embeddings
from app.storage.db import get_images_by_ids


"""
Semantic search service.
"""

# Search images by text query, limit: number of results
def semantic_search(query: str, limit: int = 3) -> list[dict]:
    
    query = query.strip()
    if not query:
        return []

    # Search in vector_db
    matched_ids = search_embeddings(query, limit=limit)

    if not matched_ids:
        return []

    image_ids = [int(image_id) for image_id in matched_ids]
    
    # Get image records as list from SQLite
    rows = get_images_by_ids(image_ids)

    # Convert list to dictionary (e.g. 3: {...})
    row_map = {row["id"]: dict(row) for row in rows}

    # Generate results
    results = []
    for image_id in image_ids:
        if image_id in row_map:
            results.append(row_map[image_id])
    
    '''results format: 
    [
    {id:3, file_name:..., caption:...},
    {id:7, file_name:..., caption:...}
    ]
    ''' 
    return results