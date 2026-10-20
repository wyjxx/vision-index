# Vision-Index

**Multimodal product search with CLIP**

## Introduction

Vision-Index lets users browse products, upload images, index images, and search with natural-language.

The current version provides a pretrained CLIP baseline, a FastAPI dashboard, and an offline evaluation pipeline for measuring future fine-tuning gains.

This project's next step goal is to explore domain adaptation for multimodal product search: fine-tuning CLIP on domain-specific query–product data to improve retrieval in a target product domain.

**Tech Stack:** Python · PyTorch · Hugging Face Transformers · FastAPI · SQLite · ChromaDB


## Demo

### Upload Images, Run Indexing

![Homepage showing the gallery](doc/homepage.png)

### Search by Queries and Inspect Scores

![Search results](doc/search.png)

---

## Key Features

- **Semantic product search:** retrieve product images using natural-language queries.
- **Product gallery:** browse a paginated catalog with images, titles, and indexing status.
- **Image uploads:** add products through the web dashboard.
- **Batch indexing:** index all pending products with one click; completed batches are skipped on subsequent runs.
- **Local inference:** run CLIP on CUDA when available, with CPU fallback.
- **Baseline evaluation:** exact full-catalog ranking with Recall@K and nDCG@K.

---

## Pipeline Architecture

### Image Indexing

```mermaid
flowchart LR
    A[Import or upload products] --> B[Images on disk + metadata in SQLite]
    B --> C[CLIP image encoder]
    C --> D[(Chroma image vectors)]
```

### Product Search

```mermaid
flowchart LR
    A[Search query] --> B[CLIP text encoder]
    B --> C[Cosine search in Chroma]
    C --> D[Load product metadata from SQLite]
    D --> E[Ranked product cards]
```

**Model:** `openai/clip-vit-base-patch32` · normalized 512-dimensional embeddings.

Implementation: [Indexing](app/services/index.py) · [Search](app/services/search.py) · [CLIP encoders](app/ai/clip.py)

---

## Dataset & Evaluation

Current experiments use a subset of [Marqo-GS-10M](https://huggingface.co/datasets/Marqo/marqo-GS-10M), from `in_domain-350.parquet`:

| Products | Queries | Query–product pairs | Train / Validation / Test |
|---:|---:|---:|---:|
| 9,934 | 200 | 10,000 | 8,000 / 1,000 / 1,000 |

Partitions are sampled by unique query–product pair. All test queries also appear in train, so this evaluates held-out pairs rather than unseen queries.

**Pretrained CLIP quality evaluation is complete:** 198 test queries ranked against all 9,934 products using exact cosine similarity.

| Metric | Test labels only | All known labels |
|---|---:|---:|
| Recall@100 | 0.6318 | 0.6435 |
| nDCG@10 | 0.0952 | 0.3746 |
| nDCG@50 | 0.2000 | 0.4451 |
| nDCG@100 | 0.2358 | 0.5301 |

Test-only scores use the 1,000 held-out pairs. All-known scores use annotations from all three splits, including training pairs, and describe catalog retrieval quality rather than held-out generalization.

**Next:** domain fine-tuning and latency/resource benchmarks. Performance measurements are not yet available.

---

## Quick Start

Use Python 3.12. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open [localhost:8000](http://localhost:8000), upload products, click **Index All Unindexed**, then search. CLIP weights are downloaded on first use.

For a prepared Marqo catalog, run `python -m scripts.import_products_sqlite` before starting the app. Dataset preparation and script usage are documented in [TECHNICAL.md](TECHNICAL.md).

For NVIDIA CUDA 12.8, install this PyTorch pair **before** the requirements command:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only wheels, replace `cu128` with `cpu`. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2-11-0).

Data lives in `data/v2/`; archived V1 data lives in `data/legacy/`. Docker packaging is included; container and GPU deployment validation is pending.

---

## Legacy V1

The original version is preserved in `app_legacy/` for general photo retrieval.

| | V1 — Photo search | V2 — Product search |
|---|---|---|
| Image representation | Qwen VLM caption → Nomic text embedding | CLIP image embedding |
| Query representation | Nomic text embedding | CLIP text embedding |
| Retrieval | Vector search + keyword reranking | Cosine vector search |

With the Python dependencies installed, Ollama running, and `qwen3.5:4b` / `nomic-embed-text:latest` available, start V1 with:

```powershell
python -m uvicorn app_legacy.main:app --reload --port 8001
```

Open [localhost:8001](http://localhost:8001). See the [Legacy README](app_legacy/README.md) for setup, screenshots, and implementation details.
