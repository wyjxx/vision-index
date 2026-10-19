"""Clean, parse, format, and combine metadata produced by the VLM."""

import json

from app_legacy.services.helper import (
    normalize_text_list,
    parse_json_list,
    format_list_for_display,
)


# Normalize attributes into the fixed storage structure.
def normalize_attributes(attributes: dict | None) -> dict:
    attributes = attributes or {}

    return {
        "lighting": normalize_text_list(attributes.get("lighting")),
        "color": normalize_text_list(attributes.get("color")),
    }


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
