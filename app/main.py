from fastapi import FastAPI, Request, UploadFile, File, Query, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import shutil

from app.config import gallery_dir, inbox_dir, thumbs_dir, search_limit
from app.services.pipeline import run_pipeline, list_inbox_images
from app.services.helper import (
    format_image_record,
    make_thumbnail,
    parse_attributes,
    parse_json_list,
)
from app.services.search import semantic_search
from app.storage.db import get_all_images, init_db, delete_image_by_id, get_image_by_path
from app.storage.vector_db import delete_embedding


'''
Main FastAPI application.

Responsibilities:
- initialize database
- provide dashboard page
- handle uploads
- run indexing pipeline
- provide search API
'''


app = FastAPI(title="vision-index")


# Initialize database on startup
@app.on_event("startup")
def on_startup() -> None:
    init_db()


# Mount folders for browser access
app.mount("/gallery", StaticFiles(directory=gallery_dir), name="gallery")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Template engine
templates = Jinja2Templates(directory="app/templates")


# Helper function: Build inbox gallery data
def build_gallery() -> list[dict]:
    '''
    Merge inbox images and indexed metadata
    into a single gallery list.
    '''

    # Get all images from inbox
    inbox_files = list_inbox_images()

    # Get all indexed image records from SQLite database
    rows = get_all_images()

    # Convert database rows into display-ready dictionaries.
    indexed_map = {row["file_name"]: format_image_record(dict(row)) for row in rows}

    images = []

    # Iteration through inbox image list (not SQLite database)
    for path in inbox_files:

        file_name = path.name
        thumb_file = thumbs_dir / file_name

        # Use thumbnail if available
        if thumb_file.exists():
            thumb_path = f"/gallery/thumbs/{file_name}"
        else:
            thumb_path = f"/gallery/inbox/{file_name}"

        image = {
            "file_name": file_name,
            "image_path": f"/gallery/inbox/{file_name}",
            "thumb_path": thumb_path,
            "indexed": False,
        }

        # Attach metadata if indexed
        if file_name in indexed_map:
            row = indexed_map[file_name]

            image.update(
                {
                    "id": row["id"],
                    "indexed": True,
                    "caption": row["caption"],
                    "objects": row["objects"],
                    "scene_tags": row["scene_tags"],
                    "attributes": row["attributes"],
                }
            )

        images.append(image)

    return images


# Dashboard page
@app.get("/")
def dashboard(request: Request, q: str = ""):
    # Build inbox gallery data
    images = build_gallery()
    
    # Return webpage
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "images": images,
            "query": q,
            "search_results": [],
        },
    )


# 1. Upload API
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Save uploaded file into inbox.
    inbox_dir.mkdir(parents=True, exist_ok=True)
    save_path = inbox_dir / file.filename

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create thumbnail
    make_thumbnail(save_path)

    return RedirectResponse("/", status_code=303)


# 2. Run pipeline API
@app.post("/run-pipeline")
def run_pipeline_route():
    run_pipeline()

    return RedirectResponse("/", status_code=303)


# 3. Search API
@app.get("/search")
def search_page(request: Request, q: str = Query(default="")):
    # Build inbox gallery data
    images = build_gallery()
    
    # Search query and format results for display.
    search_results = semantic_search(q, limit=search_limit) if q.strip() else []
    search_results = [format_image_record(row) for row in search_results]
    
    # Refresh webpage
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "images": images,
            "query": q,
            "search_results": search_results,
        },
    )


# 4. Delete API
@app.post("/delete")
def delete_image(file_name: str = Form(...)):
    # Build image absolute path
    image_path = inbox_dir / file_name
    thumb_path = thumbs_dir / file_name

    # Get image records by relative path
    row = get_image_by_path(f"inbox/{file_name}")

    # Delete original
    if image_path.exists():
        image_path.unlink()

    # Delete thumbnail
    if thumb_path.exists():
        thumb_path.unlink()

    # If indexed: delete databases
    if row:
        image_id = row["id"]
        delete_embedding(image_id)
        delete_image_by_id(image_id)
    else:
        print("NO SQLite database row matched!")

    return RedirectResponse("/", status_code=303)


# Debug API: return all indexed images
@app.get("/images")
def list_images():
    rows = get_all_images()

    # Parse JSON fields before returning them.
    return [
        {
            **dict(row),
            "objects": parse_json_list(row["objects"]),
            "scene_tags": parse_json_list(row["scene_tags"]),
            "attributes": parse_attributes(row["attributes"]),
        }
        for row in rows
    ]
