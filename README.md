# Vision-Index

`Vision-Index` is a small local image indexing project.

It scans images from `gallery/inbox`, analyzes them with a local vision-language model through Ollama, stores structured metadata in SQLite, stores embeddings in Chroma, and provides semantic image search in a simple FastAPI dashboard.

## Features

- upload images from the web UI
- scan local images from `gallery/inbox`
- generate thumbnails for gallery display
- analyze images with a local VLM via Ollama
- extract structured metadata:
  - `caption`
  - `objects`
  - `scene_tags`
  - `attributes.lighting`
  - `attributes.color`
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
  -> scan supported images
  -> generate thumbnail
  -> analyze image with Ollama VLM
  -> build structured metadata
  -> save metadata in SQLite
  -> generate embedding
  -> save embedding in Chroma
  -> search and rerank in dashboard
```

## Project Structure

```text
vision-index/
|-- app/
|   |-- ai/
|   |   `-- llm.py
|   |-- services/
|   |   |-- helper.py
|   |   |-- pipeline.py
|   |   `-- search.py
|   |-- storage/
|   |   |-- db.py
|   |   `-- vector_db.py
|   |-- static/
|   |   `-- style.css
|   |-- templates/
|   |   `-- dashboard.html
|   |-- config.py
|   `-- main.py
|-- data/
|-- evaluation/
|   |-- eval_search.py
|   |-- grid_search.py
|   |-- golden_queries.json
|   |-- grid_search_result.json
|   `-- result_v*.json
|-- gallery/
|   |-- inbox/
|   `-- thumbs/
|-- scripts/
|   |-- rename_images.py
|   `-- reset_db.py
|-- .env.example
|-- requirements.txt
`-- README.md
```

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
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Usage

### Dashboard

- upload images from the home page
- browse all files currently in `gallery/inbox`
- run the indexing pipeline from the UI
- search with natural language queries
- inspect indexed metadata in the gallery cards
- delete an image and remove its embedding at the same time

### Indexing pipeline

The pipeline:

- scans supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- skips files that already exist in SQLite
- processes new images with up to 3 worker threads
- generates thumbnails in `gallery/thumbs`
- writes metadata to `data/images.db`
- writes embeddings to `data/vector/`

### Search

Search is a two-stage flow:

1. recall candidates from Chroma using embeddings
2. rerank them with a weighted combination of semantic score and keyword matches across caption, objects, scene tags, and attributes

Current search-related settings live in `app/config.py`.

## Evaluation

Run the offline retrieval evaluation with:

```bash
python -m evaluation.eval_search
```

This reads `evaluation/golden_queries.json` and writes the latest report to `evaluation/result.json`.

Reported metrics:

- `top1_accuracy`
- `precision@5`
- `recall@5`

Grid search is used to tune rerank weights on the golden query set:

```bash
python -m evaluation.grid_search
```

## Utility Scripts

Reset SQLite and Chroma data:

```bash
python scripts/reset_db.py
```

Rename a dataset into sequential file names:

```bash
python scripts/rename_images.py
```

## Notes

- this project is local-first
- Ollama must be reachable from the app process
- `data/` and `gallery/thumbs/` are created as the app runs
- the current UI is intentionally simple and focused on indexing plus retrieval
