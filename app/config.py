"""Paths and model settings for the V2 CLIP application."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).resolve().parent.parent
app_dir = project_root / "app"
data_dir = project_root / "data" / "v2"
product_db_path = data_dir / "products.db"

clip_chroma_dir = data_dir / "clip_vector"
model_cache_dir = data_dir / "models"
product_images_dir = data_dir / "marqo" / "images"
# JSONL paths are retained only as migration/data-import sources.
product_catalog_path = data_dir / "marqo" / "products.jsonl"
uploaded_images_dir = data_dir / "uploads" / "images"
uploaded_catalog_path = data_dir / "uploads" / "products.jsonl"

clip_model = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
clip_chroma_collection = os.getenv("CLIP_COLLECTION", "vision_index_clip_products")
search_limit = 10
gallery_page_size = 24
clip_index_batch_size = 32
