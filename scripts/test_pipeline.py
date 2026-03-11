from app.storage.db import reset_db, get_all_images
from app.services.pipeline import run_pipeline
from pathlib import Path
from app.ai.llm import analyze_image

# reset database
reset_db()

# Test pipeline
result = run_pipeline()
print(result)

# Show all rows in database
rows = get_all_images()

for row in rows:
    print(dict(row))