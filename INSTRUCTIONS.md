# CODEX_INSTRUCTIONS.md

## Project

vision-index - a personal student AI project.

Goal: build a simple local image indexing system.

Images are placed into `gallery/inbox/`.  
The system should analyze them, store metadata, create embeddings, support semantic search, and keep a small evaluation loop for retrieval quality.

Workflow:

1. scan images in `gallery/inbox`
2. analyze with a local vision-language model
3. generate structured metadata
4. store metadata in SQLite
5. create and store embeddings in Chroma
6. enable semantic search
7. evaluate retrieval quality with golden queries
8. build initial agent/workflows

Development stages:

- Stage 1: image understanding using vlm
- Stage 2: embeddings and retrieval
- Stage 3: evaluation and iteration
- Stage 4: simple agent/workflows


## Coding Rules

- Keep code simple and easy to understand.
- Avoid over-engineering and unnecessary abstractions.
- Prefer plain functions and module-level variables.
- Use clear names and explicit logic.
- Keep files small but do not split them too early.
- Only add structure when it becomes necessary.


## Comments

- Use short English comments.
- Add docstrings to important functions.
- Avoid long or obvious comments.


## Project Structure

```
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

### Responsibilities

config.py  
- shared project paths
- small shared settings

main.py  
- FastAPI entry point

pipeline.py  
- image indexing workflow
- scan inbox
- call VLM
- store metadata

thumbnail.py
- generate thumbnails
- store in gallery/thumbs

search.py  
- semantic search logic

eval_search.py
- offline retrieval evaluation
- runs golden queries
- computes top1 / precision@5 / recall@5

db.py  
- SQLite and Chroma access
- metadata storage

llm.py  
- local VLM calls 
- embeddings generation


## MVP Scope

Implement only the minimal working system.

Required:

- scan `gallery/inbox`
- skip already processed images
- generate thumbnails
- analyze images with a VLM
- generate:
  - caption
  - description
  - objects
  - scene tags
- store metadata in SQLite
- store embeddings in Chroma
- support semantic search with FastAPI
- support simple offline retrieval evaluation

Not needed yet:

- OCR
- agents
- background jobs
- cloud deployment
- complex frontend
- unnecessary design patterns


## General Guidance

Prefer:

- simpler code
- fewer files
- clearer logic
- easier debugging

Do not add complexity unless it is clearly needed.
