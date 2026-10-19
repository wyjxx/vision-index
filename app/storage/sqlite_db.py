"""Store V2 product metadata in SQLite; keep image files on disk."""

import sqlite3
from contextlib import contextmanager

from app.config import product_db_path, uploaded_images_dir


@contextmanager
def connect_db():
    """Commit successful writes, roll back errors, and always close the connection."""
    product_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(product_db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    with connect_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY NOT NULL,
                title TEXT NOT NULL,
                image_path TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('marqo', 'upload')),
                indexed INTEGER NOT NULL DEFAULT 0 CHECK (indexed IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


# Convert data format for sqlite
def product_values(product: dict) -> tuple:
    """Keep IDs as strings, including unsigned IDs larger than SQLite integers."""
    product_id = product["product_id"]
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("product_id must be a nonempty string")
    return (
        product_id, product.get("title") or "", product["image_path"],
        product["source"], int(product.get("indexed", False)),
    )


# For data transfer
def import_products(products) -> int:
    """Import records in one transaction; keep existing records on repeated runs."""
    with connect_db() as connection:
        connection.executemany("""
            INSERT INTO products (product_id, title, image_path, source, indexed)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO NOTHING
        """, (product_values(product) for product in products))
        return connection.total_changes


# Upload product
def save_uploaded_product(product: dict, image_data: bytes, extension: str) -> None:
    """Save an image and insert its record; remove the image if the insert fails."""
    uploaded_images_dir.mkdir(parents=True, exist_ok=True)
    image_path = uploaded_images_dir / f"{product['product_id']}{extension}"
    with image_path.open("xb") as file:
        file.write(image_data)
    try:
        with connect_db() as connection:
            connection.execute("""
                INSERT INTO products (product_id, title, image_path, source, indexed)
                VALUES (?, ?, ?, ?, ?)
            """, product_values(product))
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


# Gallery main page
def get_products_page(page: int, page_size: int) -> tuple[list[dict], int]:
    """Show newest uploads first, followed by Marqo products in import order."""
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    with connect_db() as connection:
        total = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        rows = connection.execute("""
            SELECT product_id, title, image_path, source, indexed, created_at
            FROM products
            ORDER BY (source = 'upload') DESC,
                     CASE WHEN source = 'upload' THEN rowid END DESC, rowid ASC
            LIMIT ? OFFSET ?
        """, (page_size, (page - 1) * page_size)).fetchall()
    return [{**dict(row), "indexed": bool(row["indexed"])} for row in rows], total


# Return search result
def get_products_by_ids(product_ids: list[str]) -> dict[str, dict]:
    """Resolve vector hits to the authoritative SQLite product records."""
    if not product_ids:
        return {}
    placeholders = ",".join("?" for _ in product_ids)
    with connect_db() as connection:
        rows = connection.execute(
            f"SELECT * FROM products WHERE product_id IN ({placeholders})", product_ids,
        ).fetchall()
    return {
        row["product_id"]: {**dict(row), "indexed": bool(row["indexed"])}
        for row in rows
    }


def get_unindexed_products(limit: int) -> list[dict]:
    """Read the next batch of products still waiting for image indexing."""
    if limit < 1:
        raise ValueError("limit must be positive")
    with connect_db() as connection:
        rows = connection.execute(
            "SELECT * FROM products WHERE indexed = 0 ORDER BY rowid LIMIT ?", (limit,),
        ).fetchall()
    return [{**dict(row), "indexed": False} for row in rows]


def mark_products_indexed(product_ids: list[str]) -> None:
    """Mark a batch complete only after its vectors have been stored."""
    with connect_db() as connection:
        connection.executemany(
            "UPDATE products SET indexed = 1 WHERE product_id = ?",
            ((product_id,) for product_id in product_ids),
        )
