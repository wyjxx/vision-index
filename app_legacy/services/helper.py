"""Shared text, image, and path utilities."""

import json
from pathlib import Path
from PIL import Image

from app_legacy.config import thumbs_dir, thumbnail_size


# 1. Shared text utilities: no dependency on a model or metadata schema.

# Normalize one string or string list into a clean list.
def normalize_text_list(values: list[str] | str | None) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    cleaned = []

    for value in values:
        text = str(value).strip()
        if text:
            cleaned.append(text)

    return cleaned


# Parse JSON to python list
def parse_json_list(raw_value: list[str] | str | None) -> list[str]:
    if isinstance(raw_value, list):
        return normalize_text_list(raw_value)

    if not raw_value:
        return []

    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return normalize_text_list(raw_value)

        return normalize_text_list(parsed)

    return []


# Format one list field for UI display.
def format_list_for_display(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


# 2. Shared image/path utilities: currently use the gallery directory layout.

# Convert image to thumbnail
def make_thumbnail(image_path: Path) -> str:
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = thumbs_dir / image_path.name
    if not thumb_path.exists():
        # Resize and save thumbnail file.
        with Image.open(image_path) as img:
            img.thumbnail(thumbnail_size)
            img.save(thumb_path)

    # gallery/inbox/p1.jpg -> gallery/thumbs/p1.jpg
    return f"thumbs/{image_path.name}"


# Build RELATIVE file path for image records database
def build_db_relative_path(file_path: Path) -> str:
    return f"inbox/{file_path.name}"


