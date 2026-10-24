"""Vision-Index 2.0: product upload, browsing, search, and image serving."""

import mimetypes
from contextlib import asynccontextmanager
from filelock import Timeout

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import app_dir, product_images_dir, search_limit, gallery_page_size
from app.config import uploaded_images_dir
from app.main_helper import add_product_image_urls
from app.services.search import search_clip_products
from app.services.upload import upload_product
from app.storage.sqlite_db import connect_db, get_products_page, init_db


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


@app.get("/health")
def health():
    """Check HTTP availability and the SQLite connection."""
    with connect_db() as connection:
        connection.execute("SELECT 1").fetchone()
    return {"status": "ok"}


# Upload
@app.post("/upload")
def upload_product_page(title: str = Form(), file: UploadFile = File()):
    try:
        upload_product(title, file.file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        file.file.close()
    return RedirectResponse("/", status_code=303)


# Index
@app.post("/run-pipeline")
def run_pipeline_page():
    from app.services.index import index_unindexed_products

    try:
        index_unindexed_products()
    except Timeout as exc:
        raise HTTPException(status_code=409, detail="Product indexing is already running") from exc
    return RedirectResponse("/", status_code=303)


# Main page and Search
@app.get("/")
@app.get("/search/clip")
def dashboard(
    request: Request,
    q: str = Query(default=""),
    page: int = Query(default=1, ge=1),
):
    query = q.strip()
    total_pages = 1
    if query:
        # Search for result
        products = add_product_image_urls(request,search_clip_products(query, search_limit))
        total = len(products)
    else:
        # Main page: Get products page
        records, total = get_products_page(page, gallery_page_size)
        total_pages = max(1, (total + gallery_page_size - 1) // gallery_page_size)
        if page > total_pages:
            raise HTTPException(status_code=404, detail="Product page not found")
        products = add_product_image_urls(request, records)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "query": query,
            "products": products,
            "total_products": total,
            "page": page,
            "total_pages": total_pages,
        },
    )
