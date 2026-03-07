from pydantic import BaseModel


"""
Define data structure (model)
"""


class IndexResult(BaseModel):
    total: int
    indexed: int
    skipped: int


class ImageRecord(BaseModel):
    id: int
    file_name: str
    file_path: str
    caption: str
    description: str
    objects: str
    scene_tags: str
    embedding_id: str
    thumbnail_path: str
    created_at: str