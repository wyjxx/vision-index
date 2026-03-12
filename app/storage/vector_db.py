import chromadb

from app.config import chroma_dir, chroma_collection
from app.ai.llm import generate_embedding


"""
Chroma embedding vector storage and search
"""

client = chromadb.PersistentClient(path=str(chroma_dir))

# Get vector collection
def get_collection():
    return client.get_or_create_collection(name=chroma_collection)

# Store embedding into Chroma
def upsert_embedding(image_id: int, text: str):

    collection = get_collection()

    embedding = generate_embedding(text)

    collection.upsert(
        ids=[str(image_id)],
        embeddings=[embedding],
        documents=[text],
    )

# Search embedding from Chroma
def search_embeddings(query: str, limit: int = 3):

    collection = get_collection()

    embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit,
    )

    return results["ids"][0]