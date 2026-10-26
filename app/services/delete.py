# Delete a product from the catalog, vector index, and image storage.

from filelock import FileLock

from app.config import product_images_dir, uploaded_images_dir
from app.storage import sqlite_db
from app.storage.clip_vector_db import get_clip_collection


def delete_product(product_id: str) -> None:
    with FileLock(str(sqlite_db.product_db_path) + ".index.lock", timeout=0), sqlite_db.connect_db() as connection:
        product = connection.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
        if product is None:
            raise LookupError("Product not found")
        directory = uploaded_images_dir if product["source"] == "upload" else product_images_dir
        directory = directory.resolve()
        image = (directory / product["image_path"].removeprefix("images/")).resolve()
        if not image.is_relative_to(directory) or image == directory:
            raise ValueError("Invalid product image path")

        # Keep an image that another product uses.
        shared = connection.execute(
            "SELECT 1 FROM products WHERE image_path = ? AND source = ? AND product_id != ?",
            (product["image_path"], product["source"], product_id),
        ).fetchone()
        get_clip_collection().delete(ids=[product_id])
        if shared is None:
            image.unlink(missing_ok=True)
        connection.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
