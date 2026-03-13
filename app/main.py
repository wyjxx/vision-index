from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.models import ImageRecord, IndexResult
from app.services.pipeline import run_pipeline, scan_images
from app.services.thumbnail import make_thumbnail
from app.services.search import semantic_search
from app.storage.db import get_all_images, init_db
from app.config import gallery_dir, thumbs_dir, inbox_dir

from pathlib import Path
import shutil


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


# Return all image records
@app.get("/images", response_model=list[ImageRecord])
def list_images() -> list[dict]:
    rows = get_all_images()
    return [dict(row) for row in rows]


# 1. Upload image to inbox
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    inbox_dir.mkdir(parents=True, exist_ok=True)

    save_path = inbox_dir / file.filename

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate thumbnail after upload
    make_thumbnail(save_path)

    return RedirectResponse(url="/inbox", status_code=303)


# 2. Show inbox images via HTML
@app.get("/inbox")
def inbox_viewer(request: Request):
    files = scan_images()

    images = []
    for path in files:
        # Add thumbnail path
        thumb_path = thumbs_dir / path.name
        if thumb_path.exists():
            preview_path = f"/gallery/thumbs/{path.name}"
        else:
            preview_path = f"/gallery/inbox/{path.name}"

        images.append(
            {
                "file_name": path.name, # Image name
                "thumb_path": preview_path, # Thumbnail path
                "image_path": f"/gallery/inbox/{path.name}" # Image path
            }
        )

    return templates.TemplateResponse(
        "inbox.html",
        {
            "request": request,
            "images": images,
        },
    )


# 3. Run image pipeline and Return count result
@app.post("/index", response_model=IndexResult)
def index_images() -> dict:
    return run_pipeline()


# 4. Show all image records via HTML
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

# 5. Search images by semantic query
@app.get("/search")
def search_images(q: str = Query(default=""), limit: int = 3) -> list[dict]:
    return semantic_search(q, limit=limit)

# 6. Show search results via HTML
@app.get("/search-page")
def search_page(request: Request, q: str = "", limit: int = 3):
    images = semantic_search(q, limit=limit) if q.strip() else []

    return templates.TemplateResponse(
        "viewer.html",
        {
            "request": request,
            "images": images,
            "query": q,
            "is_search": True,
        },
    )