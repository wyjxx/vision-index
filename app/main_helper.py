"""Prepare product data for the V2 web pages."""

from fastapi import Request


def add_product_image_urls(request: Request, products: list[dict]) -> list[dict]:
    """Resolve catalog image paths for both browsing and search results."""
    results = []
    for product in products:
        image_file = product["image_path"].removeprefix("images/")
        route = "uploaded_images" if product.get("source") == "upload" else "product_images"
        image_url = str(request.url_for(route, path=image_file))
        results.append({**product, "image_url": image_url})
    return results
