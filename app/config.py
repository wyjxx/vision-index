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
load_dotenv() 

# Directory paths
project_root = Path(__file__).resolve().parent.parent

app_dir = project_root / "app" # app/

data_dir = project_root / "data" # data/
db_path = data_dir / "images.db" # data/images.db
chroma_dir = data_dir / "vector" # data/vector/

gallery_dir = project_root / "gallery" # gallery/
inbox_dir = gallery_dir / "inbox"      # gallery/inbox/
thumbs_dir = gallery_dir / "thumbs"    # gallery/thumbs/

# General settings
thumbnail_size = (512, 512)
supported_image_ext = {".jpg", ".jpeg", ".png", ".webp"}

# Ollama settings -> modify .env to change model
ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
vision_model = os.getenv("VISION_MODEL", "qwen3.5:4b")

# Embedding / vector search
embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
chroma_collection = "vision_index_images"

# Search setting
recall_limit = 10
search_limit = 5
keyword_weight = 0.2