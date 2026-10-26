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
        # Reject links to products or collections that do not exist.
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    with connect_db() as connection:
        # Store the product records.
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
        # Store the collection names.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE CHECK (length(trim(name)) > 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Link each product to its collections.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS collection_products (
                collection_id INTEGER NOT NULL,
                product_id TEXT NOT NULL,
                PRIMARY KEY (collection_id, product_id),
                FOREIGN KEY (collection_id) REFERENCES collections(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
                    ON DELETE CASCADE
            )
        """)


# List all collections and count their products.
def list_collections() -> list[dict]:
    with connect_db() as connection:
        rows = connection.execute("""
            SELECT collections.id, collections.name,
                   COUNT(collection_products.product_id) AS product_count
            FROM collections
            LEFT JOIN collection_products
                ON collections.id = collection_products.collection_id
            GROUP BY collections.id, collections.name
            ORDER BY collections.name
        """).fetchall()
    return [dict(row) for row in rows]


# Find a collection by name.
def get_collection(name: str) -> dict | None:
    if not isinstance(name, str):
        raise ValueError("collection name must be a nonempty string")
    name = name.strip().lower()
    if not name:
        raise ValueError("collection name must be a nonempty string")
    with connect_db() as connection:
        row = connection.execute(
            "SELECT id, name, created_at FROM collections WHERE name = ?",
            (name,),
        ).fetchone()
    return dict(row) if row is not None else None


# Create a collection if the name does not exist.
def create_collection(name: str) -> dict:
    """Return the collection details. Set created to True only for a new collection."""
    if not isinstance(name, str):
        raise ValueError("collection name must be a nonempty string")
    name = name.strip().lower()
    if not name:
        raise ValueError("collection name must be a nonempty string")
    with connect_db() as connection:
        cursor = connection.execute("""
            INSERT INTO collections (name) VALUES (?)
            ON CONFLICT(name) DO NOTHING
        """, (name,))
        created = cursor.rowcount == 1
        row = connection.execute(
            "SELECT id, name, created_at FROM collections WHERE name = ?",
            (name,),
        ).fetchone()
    return {**dict(row), "created": created}


# Add the products to the collection.
def add_products_to_collection(product_ids: list[str], collection_id: int) -> dict:

    if isinstance(collection_id, bool) or not isinstance(collection_id, int) or collection_id < 1:
        raise ValueError("collection_id must be a positive integer")
    if not isinstance(product_ids, list):
        raise ValueError("product_ids must be a list of nonempty strings")
    for product_id in product_ids:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_ids must contain nonempty strings")

    # Process each product ID once.
    unique_ids = list(dict.fromkeys(product_ids))
    # Save all new links in one transaction.
    with connect_db() as connection:
        collection = connection.execute(
            "SELECT id FROM collections WHERE id = ?", (collection_id,),
        ).fetchone()
        if collection is None:
            raise ValueError("collection does not exist")
        cursor = connection.executemany("""
            INSERT INTO collection_products (collection_id, product_id)
            VALUES (?, ?)
            ON CONFLICT(collection_id, product_id) DO NOTHING
        """, [(collection_id, product_id) for product_id in unique_ids])
        added_count = cursor.rowcount

    return {
        "collection_id": collection_id,
        "requested_count": len(unique_ids),
        "added_count": added_count,
        "already_present_count": len(unique_ids) - added_count,
    }


# Read the products in the collection.
def get_collection_products(collection_id: int) -> list[dict]:

    if isinstance(collection_id, bool) or not isinstance(collection_id, int) or collection_id < 1:
        raise ValueError("collection_id must be a positive integer value")

    with connect_db() as connection:
        collection = connection.execute(
            "SELECT id FROM collections WHERE id = ?", (collection_id,),
        ).fetchone()
        if collection is None:
            raise ValueError("collection does not exist")
        rows = connection.execute(
            """SELECT products.*
            FROM products JOIN collection_products
            ON products.product_id = collection_products.product_id
            WHERE collection_products.collection_id = ?
            ORDER BY products.product_id""", (collection_id,)
        ).fetchall()

    products = []
    for row in rows:
        product = dict(row)
        product["indexed"] = bool(product["indexed"])
        products.append(product)

    return products


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
