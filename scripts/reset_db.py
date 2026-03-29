from app.storage.db import reset_db
from app.storage.vector_db import reset_vector_db

print("Reset SQLite...")
reset_db()

print("Reset Chroma...")
reset_vector_db()

print("Done.")