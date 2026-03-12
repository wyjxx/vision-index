from app.storage.db import reset_db, get_all_images
from app.services.pipeline import run_pipeline
from app.storage.vector_db import search_embeddings

# Reset database
print("\nReset database...")
reset_db()


# Run indexing pipeline
print("\nRunning pipeline...")
result = run_pipeline()
print("Pipeline result:", result)


# Show rows stored in SQLite
print("\nImages stored in SQLite:")

rows = get_all_images()

for row in rows:
    row_dict = dict(row)
    print(
        row_dict["id"],
        row_dict["file_name"],
        row_dict["caption"],
        row_dict["description"],
        row_dict["objects"],
        row_dict["scene_tags"]
    )


# Test semantic search
print("\nTesting semantic search...")

query = "lake"
print("Query:", query)

results = search_embeddings(query)

print("Matched image ids:", results)