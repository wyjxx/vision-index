from app.storage.db import get_all_images
from app.storage.vector_db import upsert_embedding
from app.services.pipeline import build_embedding_text

# Rebuild embedding (vector database) without infuencing image record
print("Rebuilding embeddings...")

rows = get_all_images()

for row in rows:
    row = dict(row)

    image_id = row["id"]

    result = {
        "caption": row["caption"],
        "description": row["description"],
        "objects": eval(row["objects"]),
        "scene_tags": eval(row["scene_tags"]),
    }

    text = build_embedding_text(result)

    upsert_embedding(image_id, text)

    print(f"Reindexed image {image_id}")

print("Done.")