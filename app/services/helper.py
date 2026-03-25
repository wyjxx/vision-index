from pathlib import Path
from PIL import Image

from app.config import thumbs_dir, thumbnail_size


"""
General helper functions
"""

# Convert image to thumbnail
def make_thumbnail(image_path: Path) -> str:
    
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = thumbs_dir / image_path.name
    if not thumb_path.exists():
        with Image.open(image_path) as img:
            img.thumbnail(thumbnail_size)
            img.save(thumb_path)

    # gallery/inbox/p1.jpg -> gallery/thumbs/p1.jpg
    return f"thumbs/{image_path.name}"


# Build RELATIVE file path for image records database
def build_db_relative_path(file_path: Path) -> str:
    return f"inbox/{file_path.name}"


# Build text for embedding.
def build_embedding_text(result: dict) -> str:
    return " ".join([
        result["caption"],
        # result["description"],
        " ".join(result["objects"]),
        " ".join(result["scene_tags"]),
    ])