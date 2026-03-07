# CODEX_INSTRUCTIONS.md

## Project

vision-index — a personal student AI project.

Goal: build a simple local image indexing system.

Images are placed into `gallery/inbox/`.  
The system should analyze them, store metadata, create embeddings, and support semantic search, build basic for agentic workflow in the future.

Workflow:

1. scan images in `gallery/inbox`
2. analyze with a local vision-language model
3. generate structured metadata
4. store metadata in SQLite
5. create and store embeddings in Chroma
6. enable semantic search
7. build initial agent/workflows

Development stages:

- Stage 1: image understanding using vlm
- Stage 2: embeddings and retrieval
- Stage 3: simple agent/workflows


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
│  ├─ core.py
│  ├─ main.py
│  ├─ models.py
│  ├─ ai/llm.py
│  ├─ services/
│  │  ├─ pipeline.py
│  │  └─ search.py
│  └─ storage/db.py
├─ data/
│  └─ app.db
├─ gallery/
│  ├─ chroma/
│  ├─ inbox/
│  └─ thumbs/

```

### Responsibilities

core.py  
- shared project paths
- small shared settings

main.py  
- FastAPI entry point

pipeline.py  
- image indexing workflow

search.py  
- semantic search logic

db.py  
- SQLite and Chroma access

llm.py  
- local VLM calls and embeddings


## MVP Scope

Implement only the minimal working system.

Required:

- scan `gallery/inbox`
- skip already processed images
- analyze images with a VLM
- generate:
  - caption
  - objects
  - scene tags
- store metadata in SQLite
- store embeddings in Chroma
- support semantic search with FastAPI

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
```


