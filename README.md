# Vision-Index

**AI-powered E-commerce Product Search System**

**With Domain-Specific Fine-tuning and Evaluation**

## Introduction

Vision-Index lets users browse products, upload images, index images, and search with natural-language.

The current version provides an interactive FastAPI dashboard, CLIP projection-layer fine-tuning, and an offline evaluation pipeline comparing pretrained and fine-tuned retrieval across 3 test scenarios.

**Tech Stack:** Python · PyTorch · Hugging Face Transformers · FastAPI · SQLite · ChromaDB


## Demo

### Upload Images, Run Indexing

![Homepage showing the gallery](doc/homepage.png)

### Search by Queries and Inspect Scores

![Search results](doc/search.png)

---

## Key Features

- **Real-world E-commerce Data:** Processed 9,934 products from the Marqo-GS-10M dataset, including product images, metadata, and query–product pairs.
- **Interactive FastAPI Dashboard:** Supports natural-language product search, image gallery browsing, image uploads, and batch indexing.
- **Efficient Local AI Inference:** Uses CLIP to generate multimodal embeddings for semantic product retrieval.
- **CLIP Fine-tuning:** Trains text and image projection layers while keeping both encoders frozen.
- **Retrieval Benchmarking:** Compares pretrained and fine-tuned CLIP using full-catalog Recall@K and nDCG@K, including unseen-query tests.

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

## Fine-tuning & Evaluation

Training uses 10,000 query–product pairs from [Marqo-GS-10M](https://huggingface.co/datasets/Marqo/marqo-GS-10M), `in_domain-350.parquet`, split by unique pair into train / validation / test at 80% / 10% / 10%.

**Projection fine-tuning:** trained for five epochs; Epoch 4 was selected by validation nDCG@10. Evaluation uses exact full-catalog cosine ranking in FP32.

**Fine-tuned Epoch 4 vs. pretrained CLIP:** relative metric changes within each scenario.

| Query | Images | Domain similarity | Recall@100 | nDCG@10 | nDCG@50 | nDCG@100 |
|---|---|---|---:|---:|---:|---:|
| Seen | Unseen* | Same | +31.0% | +65.3% | +43.5% | +41.7% |
| Unseen | Unseen* | Similar | −1.8% | +19.4% | +3.0% | +3.4% |
| Unseen | Unseen* | Shifted toward women's fashion | −2.3% | −6.1% | −4.5% | −3.2% |

*Seen/unseen refers to local fine-tuning. The first test excludes training-image matches; external tests use new product IDs, with image-overlap limits detailed in [TECHNICAL.md](TECHNICAL.md#10-progress-and-next-step). Domain similarity describes product themes.*

**Next:** broaden training query coverage and validate unseen-query generalization. Latency and resource benchmarks remain pending.

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
