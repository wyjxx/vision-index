# Vision-Index

> This document describes the archived V1 system. Its entry point is now
> `app_legacy/main.py`; the V2 CLIP application starts from `app/main.py`.

`Vision-Index` is a local-first image indexing and retrieval project.

It ingests images from `data/legacy/gallery/inbox`, extracts structured visual metadata with a local vision-language model via Ollama, stores metadata in SQLite and embeddings in Chroma, and serves semantic search through a lightweight FastAPI dashboard.

## Features

- upload images from the web UI
- scan local images from `data/legacy/gallery/inbox`
- generate thumbnails for gallery display
- analyze images with a local VLM via Ollama
- extract structured metadata: `caption`, `objects`, `scene_tags`, `attributes.lighting`, `attributes.color`
- store metadata in SQLite
- store embeddings in Chroma
- search images with natural language
- rerank search results with field-aware keyword scores
- evaluate retrieval quality with golden queries
- delete images from both storage and index

## Stack

- Python
- FastAPI
- Jinja2
- SQLite
- ChromaDB
- Ollama
- Pillow

## How It Works

```text
gallery/inbox
  -> ingest new images
  -> analyze with Ollama VLM
  -> build structured metadata
  -> save metadata in SQLite
  -> generate embedding
  -> save embedding in Chroma
  -> search by natural language query
  -> rerank results
  -> display results in dashboard
```

## Project Structure


## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root.

Example:

```env
OLLAMA_HOST=http://127.0.0.1:11434
VISION_MODEL=qwen3.5:4b
EMBEDDING_MODEL=nomic-embed-text:latest
```

### 4. Start Ollama and pull models

Make sure Ollama is running and the required models are available:

```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text:latest
```

### 5. Run the app

```bash
uvicorn app_legacy.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Usage

### Dashboard

- upload images from the home page
- browse all files currently in `data/legacy/gallery/inbox`
- run the indexing pipeline from the UI
- search with natural language queries
- inspect indexed metadata in the gallery cards
- delete an image and remove its embedding at the same time

### Search

Search uses a two-stage flow:

1. recall candidates from Chroma using embeddings
2. rerank them with a weighted combination of semantic score and keyword matches across caption, objects, scene tags, and attributes

`final_score` is computed as:

```text
final_score =
  global_weight * global_score
  + caption_weight * caption_match
  + object_weight * objects_match
  + scene_weight * scene_tags_match
  + attribute_weight * attributes_match
```

- `global_score` is derived from distance between query and embedding
- `caption_match`, `objects_match`, `scene_tags_match`, and `attributes_match` are keyword match scores
- weights are configured in `app_legacy/config.py`: `global_weight`, `caption_weight`, `object_weight`, `scene_weight`, `attribute_weight`

## Evaluation

Offline evaluation:

```bash
python -m app_legacy.evaluation.eval_search
```

Metrics:

- `top1_accuracy`
- `precision@5`
- `recall@5`

Grid search for rerank weights:

```bash
python -m app_legacy.evaluation.grid_search
```

## Notes

- this project is local-first
- Ollama must be reachable from the app process
- `data/` and `gallery/thumbs/` are created as the app runs
- the current UI is intentionally simple and focused on indexing plus retrieval
