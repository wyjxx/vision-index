import base64
import json
from pathlib import Path

import requests

from app.core import ollama_host, vision_model


"""
Ollama VLM call.

Send an image to VLM and returns structured metadata.
"""


def encode_image(image_path: Path) -> str:
    """Return base64 string for one image file."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def analyze_image(image_path: Path) -> dict:
    """Analyze one image with VLM."""
    image_b64 = encode_image(image_path)

    schema = {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "description": {"type": "string"},
            "objects": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scene_tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["caption", "description", "objects", "scene_tags"],
    }

    prompt = """
    /no_think

    Analyze this image and return structured JSON.

    Rules:
    - caption: one short sentence
    - description: 1-2 sentences
    - objects: main visible objects only
    - scene_tags: short scene or style tags
    - use English
    """

    response = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": schema,
            "think": False, # disable think mode
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    #print("full response:", data)
    #print("raw response text:", data.get("response"))

    result = json.loads(data["response"])

    return {
        "caption": result.get("caption", ""),
        "description": result.get("description", ""),
        "objects": result.get("objects", []),
        "scene_tags": result.get("scene_tags", []),
    }