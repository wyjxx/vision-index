import json
import sqlite3
from pathlib import Path

from app.config import db_path

"""
Simple SQLite helpers.

This module handles:
- database connection
- table creation
- basic image record queries
"""

# Create a SQLite connection.
def connect_db() -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Delete then init db
def reset_db():
    if db_path.exists():
        db_path.unlink()
    init_db()

# Create the images table if it does not exist.
def init_db() -> None:
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                caption TEXT,
                description TEXT,
                objects TEXT,
                scene_tags TEXT,
                embedding_id TEXT,
                thumbnail_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

# Check whether an image is already indexed.
def image_exists(file_path: Path | str) -> bool:
    with connect_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM images WHERE file_path = ? LIMIT 1",
            (str(file_path),),
        ).fetchone()
    return row is not None

# Insert one image record.
def insert_image(
    file_name: str,
    file_path: str,
    caption: str = "",
    description: str = "",
    objects: list[str] | None = None,
    scene_tags: list[str] | None = None,
    embedding_id: str = "",
    thumbnail_path: str = "",
) -> int:
    objects = objects or []
    scene_tags = scene_tags or []

    with connect_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO images (
                file_name,
                file_path,
                caption,
                description,
                objects,
                scene_tags,
                embedding_id,
                thumbnail_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_name,
                file_path,
                caption,
                description,
                json.dumps(objects, ensure_ascii=False),
                json.dumps(scene_tags, ensure_ascii=False),
                embedding_id,
                thumbnail_path,
            ),
        )
        conn.commit()
        
        # Return image_id
        return cursor.lastrowid

# Return all images.
def get_all_images() -> list[sqlite3.Row]:
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                file_name,
                file_path,
                caption,
                description,
                objects,
                scene_tags,
                embedding_id,
                thumbnail_path,
                created_at
            FROM images
            ORDER BY id DESC
            """
        ).fetchall()
    return rows

# Return one image by path.
def get_image_by_path(file_path: Path | str) -> sqlite3.Row | None:
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT
                id,
                file_name,
                file_path,
                caption,
                description,
                objects,
                scene_tags,
                embedding_id,
                thumbnail_path,
                created_at
            FROM images
            WHERE file_path = ?
            """,
            (str(file_path),),
        ).fetchone()
    return row