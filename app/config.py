from pathlib import Path
import os
from dotenv import load_dotenv
"""
## Centralized project paths & settings ##

# Usage:
from app.config import inbox_dir, db_path
for img in inbox_dir.iterdir():
    print(img)

# Adding new paths example:
    ocr_dir = gallery_dir / "ocr"
    cache_dir = project_root / "cache"
"""

# Directory paths
project_root = Path(__file__).resolve().parent.parent

app_dir = project_root / "app" # app/

data_dir = project_root / "data" # data/
db_path = data_dir / "app.db" # data/app.db

gallery_dir = project_root / "gallery" # gallery/
inbox_dir = gallery_dir / "inbox"      # gallery/inbox/
thumbs_dir = gallery_dir / "thumbs"    # gallery/thumbs/
chroma_dir = gallery_dir / "chroma"    # gallery/chroma/

# General settings
thumbnail_size = (512, 512)
supported_image_ext = {".jpg", ".jpeg", ".png", ".webp"}

# Ollama settings -> modify .env to change model
ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
vision_model = os.getenv("VISION_MODEL", "qwen3.5:9b")