"""Import product JSONL metadata into V2 SQLite and initialize index status."""

import json

from filelock import FileLock

from app.config import (
    product_catalog_path, uploaded_catalog_path, product_db_path,
    clip_chroma_dir, clip_chroma_collection, clip_model,
)
from app.storage.sqlite_db import init_db, import_products, get_products_page


def existing_index_ids() -> set[str]:
    """Inspect existing vectors without loading a model or creating a collection."""
    if not (clip_chroma_dir / "chroma.sqlite3").is_file():
        return set()
    import chromadb

    client = chromadb.PersistentClient(path=str(clip_chroma_dir))
    try:
        collection = client.get_collection(clip_chroma_collection, embedding_function=None)
    except chromadb.errors.NotFoundError:
        return set()
    if (collection.metadata or {}).get("clip_model") != clip_model:
        raise ValueError("Existing collection belongs to a different CLIP model")
    return set(collection.get(include=[])["ids"])


def read_existing_products(index_ids: set[str]):
    for path, source in ((product_catalog_path, "marqo"), (uploaded_catalog_path, "upload")):
        if not path.exists():
            continue
        if source == "upload":
            with FileLock(str(path) + ".lock"):
                lines = path.read_text(encoding="utf-8").splitlines()
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            product = json.loads(line)
            yield {**product, "source": source, "indexed": product["product_id"] in index_ids}


def main():
    index_ids = existing_index_ids()
    init_db()
    added = import_products(read_existing_products(index_ids))
    _, total = get_products_page(1, 1)
    print(f"Imported {added} new products; total: {total}")
    print(f"Existing CLIP index contains {len(index_ids)} product IDs")
    print(f"SQLite: {product_db_path}")
    print("Original JSONL files, images and vectors have been retained.")


if __name__ == "__main__":
    main()
