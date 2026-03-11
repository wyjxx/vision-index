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


# Initialize database on app startup.
@app.on_event("startup")
def on_startup() -> None:
    init_db()


# Simple health check.
@app.get("/")
def read_root() -> dict:
    return {
        "message": "vision-index is running"
    }


# Run the image indexing pipeline.
@app.post("/index", response_model=IndexResult)
def index_images() -> dict:
    return run_pipeline()


# Return all indexed image records.
@app.get("/images", response_model=list[ImageRecord])
def list_images() -> list[dict]:
    rows = get_all_images()
    return [dict(row) for row in rows]