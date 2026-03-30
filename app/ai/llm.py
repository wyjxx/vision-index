import base64
import json
from pathlib import Path

import requests

from app.config import ollama_host, vision_model, embedding_model
from app.services.helper import normalize_attributes, normalize_text_list


"""
Ollama VLM call.

Send an image to VLM and returns structured metadata.
"""

# Return base64 string for one image file.
def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")

# Analyze one image with VLM and normalize the output fields.
def analyze_image(image_path: Path) -> dict:
    # Encode image before sending to Ollama.
    image_b64 = encode_image(image_path)

    # Response format schema
    schema = {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "objects": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scene_tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "attributes": {
                "type": "object",
                "properties": {
                    "lighting": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "color": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["lighting", "color"],
            },
        },
        "required": ["caption", "objects", "scene_tags", "attributes"],
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
    - objects: 4-8 main visible objects only
    - scene_tags: 3-6 short scene tags
    - attributes.lighting: visible lighting tags such as daytime, night, bright, dim, sunny, cloudy
    - attributes.color: visible main colors ordered by coverage from most dominant to least dominant
    """

    # Request structured output from the vision model.
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

    # Parse and normalize the JSON response.
    result = json.loads(data["response"])

    return {
        "caption": result.get("caption", ""),
        "objects": normalize_text_list(result.get("objects")),
        "scene_tags": normalize_text_list(result.get("scene_tags")),
        "attributes": normalize_attributes(result.get("attributes")),
    }


# Generate embedding using embedding model.
def generate_embedding(text: str) -> list[float]:
    # Request one embedding vector for the input text.
    response = requests.post(
        f"{ollama_host}/api/embeddings",
        json={
            "model": embedding_model,
            "prompt": text,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    # Return the embedding array directly.
    return data["embedding"]
