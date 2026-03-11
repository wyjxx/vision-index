# vision-index

vision-index is a small personal AI project for building a **local image indexing system**.

The goal is to explore how vision-language models can automatically analyze personal photo collections and enable structured metadata storage and semantic retrieval.

Images are placed into a local `gallery/inbox` folder.  
The system analyzes them using a local vision model, generates metadata, and stores results in a database.

---

# Features (current stage)

Current capabilities:

- scan images from `gallery/inbox`
- analyze images using a local vision-language model (Ollama)
- generate structured metadata
  - caption
  - description
  - objects
  - scene tags
- generate thumbnails
- store results in SQLite
- view indexed images in a web interface

This repository currently implements **Stage 1: image understanding**.

---

# Project structure

```

vision-index
│
├─ app
│  ├─ main.py # FastAPI entry
│  ├─ config.py # paths and configuration
│  ├─ models.py # Pydantic models
│  │
│  ├─ ai
│  │ └─ llm.py # VLM / Ollama calls
│  │
│  ├─ services
│  │ ├─ pipeline.py # indexing pipeline
│  │ ├─ thumbnail.py # thumbnail generation
│  │ └─ search.py # (future semantic search)
│  │
│  ├─ storage
│  │ └─ db.py # SQLite storage
│  │
│  └─ templates
│    └─ viewer.html # web UI
│
├─ gallery
│  ├─ inbox # new images
│  ├─ thumbs # generated thumbnails
│  └─ chroma # future vector store
│
└─ scripts
└─ test_pipeline.py

```

---

# Pipeline

The indexing workflow:

```

gallery/inbox
↓
scan images
↓
vision model analysis
↓
generate metadata
↓
create thumbnail
↓
store in SQLite
↓
view in web UI

```

---

# Running the project

### 1 Install dependencies

```

pip install -r requirements.txt

```

### 2 Start Ollama and the vision model

Example:

```

ollama run qwen2.5vl

```

### 3 Run the server

```

uvicorn app.main:app --reload

```

### 4 Open the viewer

```

[http://localhost:8000/viewer](http://localhost:8000/viewer)

```

---

# Development stages

The project is developed incrementally.

### Stage 1
Image understanding

- VLM image analysis
- metadata generation
- SQLite storage
- web viewer

### Stage 2
Semantic retrieval

- embedding generation
- vector database (Chroma)
- text-to-image search

### Stage 3
Agent workflows

- natural language search
- automated organization
- AI-assisted photo workflows

---

# Purpose

This project is mainly a **learning experiment** for exploring:

- vision-language models
- local AI pipelines
- semantic indexing
- lightweight AI applications