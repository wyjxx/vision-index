"""Vision-Index 2.0: product upload, browsing, search, and image serving."""

import mimetypes
from contextlib import asynccontextmanager
from filelock import Timeout

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent.workflow import (
    ConversationNotFoundError, ConversationNotWaitingError,
    resume_agent as resume_workflow, run_agent,
)
from app.config import app_dir, product_images_dir, uploaded_images_dir, search_limit, gallery_page_size
from app.main_helper import add_product_image_urls
from app.services.delete import delete_product
from app.services.search import search_clip_products
from app.services.upload import upload_product
from app.storage.sqlite_db import (
    connect_db, get_collection_products, get_products_page, init_db, list_collections,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Vision-Index 2.0", version="2.0", lifespan=lifespan)

mimetypes.add_type("image/webp", ".webp")
app.mount(
    "/product-images",
    StaticFiles(directory=product_images_dir, check_dir=False),
    name="product_images",
)
app.mount("/static", StaticFiles(directory=app_dir / "static"), name="static")
app.mount(
    "/uploaded-images",
    StaticFiles(directory=uploaded_images_dir, check_dir=False),
    name="uploaded_images",
)
templates = Jinja2Templates(directory=app_dir / "templates")


# Health Check
@app.get("/health")
def health():
    """Check HTTP availability and the SQLite connection."""
    with connect_db() as connection:
        connection.execute("SELECT 1").fetchone()
    return {"status": "ok"}


# Return collection.
@app.get("/collections", tags=["Collections"])
def collections() -> list[dict]:
    return list_collections()


# Delete
@app.delete("/products/{product_id}", tags=["Products"])
def delete_product_page(product_id: str) -> dict:
    try:
        delete_product(product_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Timeout as exc:
        raise HTTPException(status_code=409, detail="Indexing is running. Try again when it finishes.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"deleted": product_id}


# Upload
@app.post("/upload", tags=["Products"])
def upload_product_page(title: str = Form(), file: UploadFile = File()) -> dict:
    try:
        return upload_product(title, file.file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        file.file.close()


# Index
@app.post("/run-pipeline", tags=["Products"])
def run_pipeline_page() -> dict:
    from app.services.index import index_unindexed_products

    try:
        count = index_unindexed_products()
    except Timeout as exc:
        raise HTTPException(status_code=409, detail="Product indexing is already running") from exc
    return {"indexed": count}


# Start agent conversation.
@app.post("/agent/start", tags=["Agent"])
def start_agent(question: str = Body(..., embed=True, min_length=1, strict=True)) -> dict:
    try:
        return run_agent(question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Continue agent conversation.
@app.post("/agent/{thread_id}/resume", tags=["Agent"])
def resume_agent(
    thread_id: str,
    reply: str = Body(..., embed=True, min_length=1, strict=True,
                      description="During chat, enter a message or q to quit. During write approval, enter y to confirm; other text cancels."),
) -> dict:
    try:
        return resume_workflow(thread_id, reply)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationNotWaitingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Main page and Search
@app.get("/", tags=["Dashboard"])
@app.get("/search/clip", tags=["Dashboard"])
def dashboard(
    request: Request,
    q: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    collection_id: int | None = Query(default=None, ge=1),
):
    query = q.strip()
    total_pages = 1
    collection = None
    if collection_id is not None:
        collection = next((item for item in list_collections() if item["id"] == collection_id), None)
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")
        query = ""
        records = get_collection_products(collection_id)
        total = len(records)
        total_pages = max(1, (total + gallery_page_size - 1) // gallery_page_size)
        records = records[(page - 1) * gallery_page_size:page * gallery_page_size]
        products = add_product_image_urls(request, records)
    elif query:
        # Search for result
        products = add_product_image_urls(request,search_clip_products(query, search_limit))
        total = len(products)
    else:
        # Main page: Get products page
        records, total = get_products_page(page, gallery_page_size)
        total_pages = max(1, (total + gallery_page_size - 1) // gallery_page_size)
        products = add_product_image_urls(request, records)
    if page > total_pages:
        raise HTTPException(status_code=404, detail="Product page not found")
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "query": query,
            "products": products,
            "total_products": total,
            "page": page,
            "total_pages": total_pages,
            "collection": collection,
            "asset_versions": {
                name: (app_dir / "static" / name).stat().st_mtime_ns
                for name in ("style.css", "chat.js", "gallery.js", "upload.js", "index.js", "collections.js")
            },
        },
    )
