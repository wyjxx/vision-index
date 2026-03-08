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


def connect_db() -> sqlite3.Connection:
    """Create a SQLite connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def reset_db():
    '''Delete then init db'''
    if db_path.exists():
        db_path.unlink()
    init_db()

def init_db() -> None:
    """Create the images table if it does not exist."""
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


def image_exists(file_path: Path | str) -> bool:
    """Check whether an image is already indexed."""
    with connect_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM images WHERE file_path = ? LIMIT 1",
            (str(file_path),),
        ).fetchone()
    return row is not None


def insert_image(
    file_name: str,
    file_path: str,
    caption: str = "",
    description: str = "",
    objects: list[str] | None = None,
    scene_tags: list[str] | None = None,
    embedding_id: str = "",
    thumbnail_path: str = "",
) -> None:
    """Insert one image record."""
    objects = objects or []
    scene_tags = scene_tags or []

    with connect_db() as conn:
        conn.execute(
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


def get_all_images() -> list[sqlite3.Row]:
    """Return all images."""
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


def get_image_by_path(file_path: Path | str) -> sqlite3.Row | None:
    """Return one image by path."""
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