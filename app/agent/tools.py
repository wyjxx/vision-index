"""Provide tools to find products and collections."""

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app.storage.sqlite_db import get_collection as find_collection
from app.storage.sqlite_db import get_collection_products as read_collection_products
from app.storage.sqlite_db import create_collection as insert_collection
from app.storage.sqlite_db import add_products_to_collection as add_collection_products


# Pydantic model
class SearchProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, strict=True, description="Describe the products to find.",
    )
    limit: int = Field(
        default=10, ge=1, le=50, strict=True,
        description="Set the maximum number of results. Use a value from 1 to 50.",
    )

class CollectionNameInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1, strict=True,
        description="Enter the collection name. The search ignores letter case and spaces at each end.",
    )


class ReadCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_id: int = Field(
        gt=0, strict=True, description="Enter the collection ID from get_collection.",
    )
    limit: int = Field(
        default=20, ge=1, le=50, strict=True,
        description="Set the maximum number of products to return. Use 1 to 50.",
    )


class AddProductsToCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_ids: list[str] = Field(min_length=1, description="Use product IDs from search results.")
    collection_id: int = Field(gt=0, strict=True, description="Use an existing collection ID.")

# Read Tool: Search for product by query.
@tool(
    args_schema=SearchProductsInput,
    description=(
        "Search for products by query. "
        "Return candidate products number, product IDs, titles, similarity scores. "
    ),
)
def search_products(query: str, limit: int = 10) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("query must be a nonempty string")

    from app.services.search import search_clip_products

    products = search_clip_products(query, limit)
    candidates = [
        {
            "product_id": product["product_id"],
            "title": product["title"],
            "similarity": float(product["similarity"]),
        }
        for product in products
    ]
    return {"query": query, "count": len(candidates), "products": candidates}


# Read Tool: Find a collection by name.
@tool(
    args_schema=CollectionNameInput,
    description="Find a collection by name. Return its details and whether it exists.",
)
def get_collection(name: str) -> dict:
    collection = find_collection(name)
    return {"exists": collection is not None, "collection": collection}


# Read Tool: Read the products in an existing collection.
@tool(
    args_schema=ReadCollectionInput,
    description="Read products in a collection. Use get_collection to get its ID first.",
)
def get_collection_products(collection_id: int, limit: int = 20) -> dict:
    products = read_collection_products(collection_id)
    return {
        "collection_id": collection_id,
        "total_count": len(products),
        "products": [
            {"product_id": product["product_id"], "title": product["title"]}
            for product in products[:limit]
        ],
    }


# Write Tool: Create a collection
@tool(args_schema=CollectionNameInput, description="Create a collection. Requires user approval.")
def create_collection(name: str) -> dict:
    return insert_collection(name)


# Write Tool: Add products to collection
@tool(args_schema=AddProductsToCollectionInput, description=(
    "Add products to an existing collection. Requires user approval. "
    "Use product IDs from search results and a collection ID from get_collection or create_collection."
))
def add_products_to_collection(product_ids: list[str], collection_id: int) -> dict:
    return add_collection_products(product_ids, collection_id)


# Give the model tools
READ_ONLY_TOOLS = [search_products, get_collection,get_collection_products]
WRITE_TOOLS = [create_collection, add_products_to_collection]
ALL_AGENT_TOOLS = READ_ONLY_TOOLS + WRITE_TOOLS
