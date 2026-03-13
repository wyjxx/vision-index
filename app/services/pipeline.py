import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from app.ai.llm import analyze_image
from app.config import inbox_dir, supported_image_ext
from app.services.thumbnail import make_thumbnail
from app.storage.db import image_exists, insert_image
from app.storage.vector_db import upsert_embedding

"""
Image indexing pipeline.
"""

# Return all valid images in inbox.
def list_inbox_images() -> list[Path]:
    if not inbox_dir.exists():
        return []

    image_files = []

    for file_path in inbox_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_image_ext:
            image_files.append(file_path)

    return image_files


# Scan inbox and return images that are not yet indexed.
def scan_images() -> tuple[list[Path], int, int]:
    if not inbox_dir.exists():
        return [], 0, 0

    new_images = []
    indexed = 0
    skipped = 0

    for file_path in inbox_dir.iterdir():

        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_image_ext:
            continue

        db_file_path = build_db_file_path(file_path)

        if image_exists(db_file_path):
            skipped += 1
        else:
            new_images.append(file_path)
            indexed += 1

    return new_images, indexed, skipped


# Build database file path for one inbox image.
def build_db_file_path(file_path: Path) -> str:
    return f"inbox/{file_path.name}"


# Store one indexed image into SQLite.
def store_image_record(file_path: Path, result: dict, thumb_path: str) -> int:
    db_file_path = build_db_file_path(file_path)

    return insert_image(
        file_name=file_path.name,
        file_path=db_file_path,
        caption=result["caption"],
        description=result["description"],
        objects=result["objects"],
        scene_tags=result["scene_tags"],
        embedding_id="",
        thumbnail_path=thumb_path,
    )


# Build text for embedding.
def build_embedding_text(result: dict) -> str:
    return " ".join([
        result["caption"],
        # result["description"],
        " ".join(result["objects"]),
        " ".join(result["scene_tags"]),
    ])


# Index one image.
def index_image(file_path: Path) -> None:

    # Generate thumbnail
    thumb_path = make_thumbnail(file_path)

    # Timing start
    start = time.time()

    # Call VLM
    result = analyze_image(file_path)

    # Timing end
    elapsed = time.time() - start
    print(f"{file_path.name} analyzed in {elapsed:.2f}s")

    # Store image record
    image_id = store_image_record(file_path, result, thumb_path)

    # Build embedding text
    text = build_embedding_text(result)

    # Store embedding
    upsert_embedding(image_id, text)


# Scan inbox -> index images -> count results
def run_pipeline() -> dict:

    # Scan inbox
    new_images, indexed, skipped = scan_images()

    total = indexed + skipped

    # Parallel indexing
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(index_image, new_images)

    # Return count results
    return {
        "total": total,
        "indexed": indexed,
        "skipped": skipped,
    }