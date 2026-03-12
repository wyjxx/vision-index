from pathlib import Path
from PIL import Image

from app.config import thumbs_dir, thumbnail_size


"""
Generate thumbnail for one image.
"""

# Convert image to thumbnail
# gallery/inbox/p1.jpg -> gallery/thumbs/p1.jpg
def make_thumbnail(image_path: Path) -> str:
    
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = thumbs_dir / image_path.name
    if not thumb_path.exists():
        with Image.open(image_path) as img:
            img.thumbnail(thumbnail_size)
            img.save(thumb_path)

    return f"thumbs/{image_path.name}"