from fastapi import FastAPI

from app.models import ImageRecord, IndexResult
from app.services.pipeline import run_pipeline
from app.storage.db import get_all_images, init_db


"""
Minimal FastAPI app.

This module handles:
- app startup
- database initialization
- basic API routes
"""


app = FastAPI(title="vision-index")


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database on app startup."""
    init_db()


@app.get("/")
def read_root() -> dict:
    """Simple health check."""
    return {
        "message": "vision-index is running"
    }


@app.post("/index", response_model=IndexResult)
def index_images() -> dict:
    """Run the image indexing pipeline."""
    return run_pipeline()


@app.get("/images", response_model=list[ImageRecord])
def list_images() -> list[dict]:
    """Return all indexed image records."""
    rows = get_all_images()
    return [dict(row) for row in rows]