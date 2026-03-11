import base64
import json
from pathlib import Path

import requests

from app.config import ollama_host, vision_model


"""
Ollama VLM call.

Send an image to VLM and returns structured metadata.
"""

# Return base64 string for one image file.
def encode_image(image_path: Path) -> str:
    
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")

# Analyze one image with VLM
def analyze_image(image_path: Path) -> dict:
    
    image_b64 = encode_image(image_path)

    # Response format schema
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
    - Be concise, literal, and grounded in visible content
    - Prefer stable, searchable terms over expressive or poetic wording.
    - Do not guess exact location, landmark or country unless visually certain.
    - Use English

    Fields:
    - caption: one short sentence summarizing image
    - description: 1-2 short concise sentences describing the overall scene
    - objects: 4-8 main visible objects only
    - scene_tags: 3-6 short scene or style tags
    """

    response = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": schema,
            "think": False, # Disable think mode
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    # Raw model response
    #print("full response:", data)
    #print("raw response text:", data.get("response"))
    #print("MODEL:", vision_model)

    result = json.loads(data["response"])

    return {
        "caption": result.get("caption", ""),
        "description": result.get("description", ""),
        "objects": result.get("objects", []),
        "scene_tags": result.get("scene_tags", []),
    }