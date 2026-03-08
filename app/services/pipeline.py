from pathlib import Path

from app.ai.llm import analyze_image
from app.config import inbox_dir, supported_image_ext
from app.storage.db import image_exists, insert_image


"""
Image indexing pipeline.
"""


def scan_images() -> list[Path]:
    """Return all supported image files in inbox."""
    if not inbox_dir.exists():
        return []

    image_files = []

    for file_path in inbox_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_image_ext:
            image_files.append(file_path)

    return image_files


def index_image(file_path: Path) -> bool:
    """Index one image if it is not already indexed."""
    if image_exists(file_path):
        return False

    # call llm
    result = analyze_image(file_path)

    insert_image(
        file_name=file_path.name,
        file_path=str(file_path),
        caption=result["caption"],
        description=result["description"],
        objects=result["objects"],
        scene_tags=result["scene_tags"],
        embedding_id="",
        thumbnail_path="",
    )
    return True


def run_pipeline() -> dict:
    """Scan inbox and index new images -> summarize results"""
    image_files = scan_images()

    total = len(image_files)
    indexed = 0
    skipped = 0

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