"""Validate an uploaded product and ask storage to save it."""

from io import BytesIO
from uuid import uuid4

from PIL import Image

from app.storage.sqlite_db import save_uploaded_product


def upload_product(title: str, image_data: bytes) -> dict:
    """Create an unindexed product with a generated ID and verified image."""
    title = title.strip()
    if not title:
        raise ValueError("Product title must not be blank")
    try:
        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format
            image.verify()
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError("Upload a valid image") from exc
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format)
    if extension is None:
        raise ValueError("Use a JPEG, PNG or WebP image")

    product_id = f"upload_{uuid4().hex}"
    product = {
        "product_id": product_id,
        "title": title,
        "image_path": f"images/{product_id}{extension}",
        "source": "upload",
        "indexed": False,
    }
    save_uploaded_product(product, image_data, extension)
    return product
