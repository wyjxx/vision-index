import json
from pathlib import Path
from PIL import Image

from app.config import thumbs_dir, thumbnail_size


"""
General helper functions
"""

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


# Normalize attributes into the fixed storage structure.
def normalize_attributes(attributes: dict | None) -> dict:
    attributes = attributes or {}

    return {
        "lighting": normalize_text_list(attributes.get("lighting")),
        "color": normalize_text_list(attributes.get("color")),
    }


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


# Parse attributes JSON to dictionary
def parse_attributes(raw_value: dict | str | None) -> dict:
    if isinstance(raw_value, dict):
        return normalize_attributes(raw_value)

    if not raw_value:
        return normalize_attributes({})

    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return normalize_attributes({})

        return normalize_attributes(parsed)

    return normalize_attributes({})


# Format one list field for UI display.
def format_list_for_display(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


# Format attributes into one compact string for UI display.
def format_attributes_for_display(attributes: dict) -> str:
    lighting = format_list_for_display(attributes.get("lighting", []))
    color = format_list_for_display(attributes.get("color", []))
    return f"lighting: {lighting} | color: {color}"


# Format one image record into UI-friendly values.
def format_image_record(row: dict) -> dict:
    # Parse JSON metadata fields from SQLite.
    objects = parse_json_list(row.get("objects"))
    scene_tags = parse_json_list(row.get("scene_tags"))
    attributes = parse_attributes(row.get("attributes"))

    formatted = dict(row)
    formatted["objects"] = format_list_for_display(objects)
    formatted["scene_tags"] = format_list_for_display(scene_tags)
    formatted["attributes"] = format_attributes_for_display(attributes)

    return formatted


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


# Build text for embedding.
def build_embedding_text(result: dict) -> str:
    # Merge searchable metadata fields into one text block.
    attributes = normalize_attributes(result.get("attributes"))

    return " ".join([
        result.get("caption", ""),
        " ".join(normalize_text_list(result.get("objects"))),
        " ".join(normalize_text_list(result.get("scene_tags"))),
        " ".join(attributes["lighting"]),
        " ".join(attributes["color"]),
    ])
