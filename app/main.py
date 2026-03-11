from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.models import ImageRecord, IndexResult
from app.services.pipeline import run_pipeline
from app.storage.db import get_all_images, init_db
from app.config import gallery_dir

from pathlib import Path


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


# Mount folders to allow browser access
app.mount("/gallery", StaticFiles(directory=gallery_dir), name="gallery")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Create jinja2 template object
templates = Jinja2Templates(directory="app/templates")

# Simple health check.
@app.get("/")
def read_root() -> dict:
    return {
        "message": "vision-index is running"
    }


# Run image pipeline and Return count result
@app.post("/index", response_model=IndexResult)
def index_images() -> dict:
    return run_pipeline()


# Return all image records
@app.get("/images", response_model=list[ImageRecord])
def list_images() -> list[dict]:
    rows = get_all_images()
    return [dict(row) for row in rows]



# Show all image records via HTML
@app.get("/viewer")
def image_viewer(request: Request):

    # Load image data from database
    rows = get_all_images()
    images = [dict(row) for row in rows]

    # Load image data into html template and return
    return templates.TemplateResponse(
        "viewer.html",
        {
            "request": request,
            "images": images
        }
    )