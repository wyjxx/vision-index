import time
from pathlib import Path

from app.ai.llm import analyze_image
from app.config import inbox_dir, supported_image_ext
from app.storage.db import image_exists, insert_image
from app.services.thumbnail import make_thumbnail


"""
Image indexing pipeline.
"""

# Scan and return all images in inbox.
def scan_images() -> list[Path]:
    
    if not inbox_dir.exists():
        return []

    image_files = []

    # Check if image is valid format
    for file_path in inbox_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_image_ext:
            image_files.append(file_path)

    return image_files


# Index one image if it is not already indexed.
def index_image(file_path: Path) -> bool:
    
    db_file_path = f"inbox/{file_path.name}"
    # Check if image record exists
    if image_exists(db_file_path):
        return False

    # timing start
    start = time.time()

    # Generate thumbnail 
    thumb_path = make_thumbnail(file_path)
    # Call VLM
    result = analyze_image(file_path)
    
    # timing end
    elapsed = time.time() - start
    print(f"{file_path.name} analyzed in {elapsed:.2f}s")

    # Insert image record into database
    insert_image(
        file_name=file_path.name,
        file_path=f"inbox/{file_path.name}",
        caption=result["caption"],
        description=result["description"],
        objects=result["objects"],
        scene_tags=result["scene_tags"],
        embedding_id="",
        thumbnail_path=thumb_path,
    )
    return True


# Scan inbox and index new images
# Return count results
def run_pipeline() -> dict:
    
    image_files = scan_images()

    total = len(image_files)
    indexed = 0
    skipped = 0

    # Count results
    for file_path in image_files:
        if index_image(file_path):
            indexed += 1
        else:
            skipped += 1

    return {
        "total": total,
        "indexed": indexed,
        "skipped": skipped,
    }