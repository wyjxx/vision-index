# vision-index

vision-index is a small local AI project for indexing personal images.

It scans images from `gallery/inbox`, analyzes them with a local vision-language model, stores metadata in SQLite, stores embeddings in Chroma, and supports semantic image search through a simple FastAPI dashboard.

## Features

- upload images from the web UI
- scan local images from `gallery/inbox`
- analyze images with a local VLM via Ollama
- generate structured metadata:
  - caption
  - objects
  - scene tags
  - attributes
- generate thumbnails
- store metadata in SQLite
- store embeddings in Chroma
- search images with natural language
- evaluate retrieval quality with golden queries
- view all images and search results in a simple dashboard

## Stack

- Python
- FastAPI
- Jinja2
- SQLite
- Chroma
- Ollama

## Project structure

```text
vision-index/
├─ app/
│  ├─ ai/
│  │  └─ llm.py
│  ├─ services/
│  │  ├─ pipeline.py
│  │  ├─ search.py
│  │  └─ helper.py
│  ├─ storage/
│  │  ├─ db.py
│  │  └─ vector_db.py
│  ├─ templates/
│  │  └─ dashboard.html
│  ├─ static/
│  ├─ config.py
│  └─ main.py
├─ data/
│  ├─ images.db
│  └─ vector/
├─ gallery/
│  ├─ inbox/
│  └─ thumbs/
├─ evaluation/
│  ├─ eval_search.py
│  ├─ golden_queries.json
│  └─ result_v1.json
├─ scripts/
│  ├─ reset.py
│  └─ test_pipeline.py
├─ .env.example
├─ requirements.txt
└─ README.md
```

## Pipeline

```text
gallery/inbox
→
scan images
→
generate thumbnail
→
analyze image with VLM
→
generate metadata
→
store metadata in SQLite
→
generate embedding
→
store embedding in Chroma
→
semantic search in dashboard
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file if needed.

Example values:

```env
OLLAMA_HOST=http://127.0.0.1:11434
VISION_MODEL=qwen3.5:4b
EMBEDDING_MODEL=nomic-embed-text:latest
```

### 3. Start Ollama

Make sure both the vision model and embedding model are available in Ollama.
```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text:latest
```

### 4. Run the app

```bash
uvicorn app.main:app --reload
```

### 5. Open in browser

```text
http://localhost:8000
```

## Current scope

This project is intentionally kept simple:

* local-first
* small codebase
* easy to read
* easy to extend later

Current focus:

* image understanding
* metadata indexing
* embedding-based retrieval
* basic retrieval evaluation

Future direction:

* better search quality
* richer metadata
* lightweight agent workflows
