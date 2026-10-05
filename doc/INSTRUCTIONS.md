# CODEX_INSTRUCTIONS.md

## Project

vision-index - a local-first image indexing and retrieval project.

Goal: build a simple local image indexing system for images placed in `gallery/inbox/`.

The system should:

1. ingest images from `gallery/inbox/`
2. analyze them with a local vision-language model via Ollama
3. generate structured metadata
4. store metadata in SQLite
5. create and store embeddings in Chroma
6. support semantic search
7. rerank search results with field-aware keyword scores
8. evaluate retrieval quality with golden queries
9. provide a lightweight FastAPI dashboard

Development stages:

- Stage 1: image understanding with local VLM
- Stage 2: embeddings and retrieval
- Stage 3: reranking and evaluation
- Stage 4: simple UI and workflow improvements


## Coding Rules

- Keep code simple and easy to understand.
- Avoid over-engineering and unnecessary abstractions.
- Prefer plain functions and explicit logic.
- Use clear names and predictable data flow.
- Keep files small, but do not split them too early.
- Only add structure when it becomes necessary.


## Comments

- Use short English comments.
- Add docstrings to important functions.
- Avoid long or obvious comments.


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

### Responsibilities

`app/config.py`
- shared project paths
- small shared settings
- search rerank weights

`app/main.py`
- FastAPI entry point
- dashboard routes
- upload, pipeline, search, and delete endpoints

`app/services/pipeline.py`
- image indexing workflow
- scan inbox
- call VLM
- build structured metadata
- write metadata to storage

`app/services/search.py`
- semantic retrieval
- result reranking
- final search scoring

`app/services/helper.py`
- formatting helpers
- thumbnail generation
- JSON field parsing

`app/storage/db.py`
- SQLite access
- metadata storage and retrieval

`app/storage/vector_db.py`
- Chroma access
- embedding storage and deletion

`app/ai/llm.py`
- local VLM calls via Ollama
- embedding generation

`evaluation/eval_search.py`
- offline retrieval evaluation
- metrics such as top1, precision@5, recall@5

`evaluation/grid_search.py`
- rerank weight search
- evaluation over candidate weight combinations


## MVP Scope

Implement only the minimal working system.

Required:

- scan `gallery/inbox`
- upload images from the dashboard
- skip already processed images
- generate thumbnails
- analyze images with a local VLM via Ollama
- generate:
  - `caption`
  - `objects`
  - `scene_tags`
  - `attributes.lighting`
  - `attributes.color`
- store metadata in SQLite
- store embeddings in Chroma
- support semantic search with FastAPI
- rerank search results with field-aware keyword scores
- support simple offline retrieval evaluation
- allow image deletion from file storage and index

Not needed yet:

- OCR
- agents
- background jobs
- cloud deployment
- complex frontend
- unnecessary design patterns


## Search Guidance

Search should use a two-stage flow:

1. recall candidates from Chroma using embeddings
2. rerank them with semantic score plus keyword matches across metadata fields

Expected scoring fields:

- `caption`
- `objects`
- `scene_tags`
- `attributes`

Weights should stay configurable in `app/config.py`.


## General Guidance

Prefer:

- simpler code
- fewer files
- clearer logic
- easier debugging

Do not add complexity unless it is clearly needed.
