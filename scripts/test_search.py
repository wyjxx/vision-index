from app.storage.db import get_all_images
from app.storage.vector_db import search_embeddings

query = "nature"
print("Query:", query)

results = search_embeddings(query)

# Load all images from SQLite
rows = get_all_images()

# Build id → image mapping
images = {str(row["id"]): row for row in rows}

print("\nMatched results:")

for image_id in results:
    img = images.get(image_id)

    if img:
        print(
            f"id={image_id} | file={img['file_name']} | caption={img['caption']}"
        )