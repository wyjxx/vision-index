from app.storage.db import reset_db, get_all_images
from app.services.pipeline import run_pipeline
from pathlib import Path
from app.ai.llm import analyze_image

reset_db()

# test pipeline
result = run_pipeline()
print(result)

rows = get_all_images()
for row in rows:
    print(dict(row))
