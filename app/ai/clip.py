"""CLIP image and text encoders."""

from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import clip_model, model_cache_dir


@lru_cache(maxsize=1)
def load_clip() -> tuple[CLIPProcessor, CLIPModel, torch.device]:
    """Load one shared CLIP instance on the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=model_cache_dir)
    model = CLIPModel.from_pretrained(
        clip_model,
        cache_dir=model_cache_dir,
        dtype=dtype,
    ).to(device).eval()
    return processor, model, device


def encode_images(image_paths: list[Path | str]) -> list[list[float]]:
    """Encode image files as normalized 512-dimensional CLIP vectors."""
    if not image_paths:
        return []

    processor, model, device = load_clip()
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)
    with torch.inference_mode():
        output = model.get_image_features(pixel_values=pixel_values)
        vectors = output.pooler_output
        vectors = F.normalize(vectors, dim=-1)
    return vectors.float().cpu().tolist()


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Encode text queries as normalized 512-dimensional CLIP vectors."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("CLIP texts must be nonempty strings")

    processor, model, device = load_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.get_text_features(**text_inputs)
        vectors = output.pooler_output
        vectors = F.normalize(vectors, dim=-1)
    return vectors.float().cpu().tolist()
