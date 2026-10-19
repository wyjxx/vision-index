"""Coordinate CLIP image encoding and vector storage for indexing."""

from pathlib import Path

from filelock import FileLock

from app.ai.clip import encode_images
from app.config import product_images_dir, uploaded_images_dir, product_db_path, clip_index_batch_size
from app.storage.clip_vector_db import upsert_clip_embeddings
from app.storage.sqlite_db import get_unindexed_products, mark_products_indexed


def index_clip_products(products: list[dict], dataset_dir: Path | str | None = None) -> int:
    """Encode one batch using either an explicit dataset or each product's source."""
    if not products:
        return 0
    image_paths = []
    for product in products:
        if dataset_dir is not None:
            image_path = Path(dataset_dir) / product["image_path"]
        else:
            directory = uploaded_images_dir if product["source"] == "upload" else product_images_dir
            image_path = directory / product["image_path"].removeprefix("images/")
        image_paths.append(image_path)
    embeddings = encode_images(image_paths)
    return upsert_clip_embeddings(products, embeddings)


def index_unindexed_products(batch_size: int = clip_index_batch_size) -> int:
    """Process all pending products; retain successful batches if a later one fails."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    processed = 0
    # Only one bulk indexing job may use the GPU at a time.
    with FileLock(str(product_db_path) + ".index.lock", timeout=0):
        while True:
            products = get_unindexed_products(batch_size)
            if not products:
                break
            count = index_clip_products(products)
            mark_products_indexed([product["product_id"] for product in products])
            processed += count
            print(f"Indexed {processed} pending products", flush=True)
    return processed
