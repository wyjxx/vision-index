from app.storage.db import reset_db, get_all_images
from app.services.pipeline import run_pipeline
from app.storage.vector_db import search_embeddings
import time

# Reset database
print("\nReset database...")
reset_db()