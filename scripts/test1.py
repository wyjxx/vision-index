from app.storage.db import reset_db, get_all_images
from app.services.pipeline import run_pipeline
from pathlib import Path
from app.ai.llm import analyze_image

reset_db()

# test vlm
result = analyze_image(Path("gallery/inbox/p1.jpg"))
print(result)