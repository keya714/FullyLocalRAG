# Fully-Local RAG (CLI) — Laptop-Ready

A Retrieval-Augmented Generation (RAG) system that runs **entirely on a laptop**, no cloud calls.  
- **LLM**: local `llama.cpp` with a quantized **Phi-3-mini-instruct (Q4 GGUF)** (configurable).  
- **Retrieval**: **Hybrid** (Dense FAISS + BM25) with optional **cross-encoder reranking**.  
- **CLI**: ask questions from the terminal; prints answer + sources.  
- **Guardrails**: fast keyword deny-list (configurable).  
- **Docs pipeline**: clean PDF → chunk → embed → FAISS index.

---

## 1) Features mapped to the Challenge Requirements

### Language Model
- **Pre-trained, quantized model**: uses a **GGUF** quantized model (`Phi-3-mini-4k-instruct-q4.gguf`) via `llama.cpp`.  
- **Quantization**: Q4 (fast + low memory). Swap models through `config.yml`.  
- **Local inference**: runs CPU-only; threads configurable; optional GPU layers if available.

### Retrieval Mechanism
- **Corpus creation**: ingest PDFs from `./corpus/` (configurable).  
- **Chunking**: `RecursiveCharacterTextSplitter` with overlap; basic cleaning to reduce noise.  
- **Embeddings**: Sentence Transformers (HuggingFace) — device configurable (CPU/GPU).  
- **Index**: FAISS saved to disk (`./index`).  
- **Hybrid retrieval**: dense FAISS + **BM25** (via `rank_bm25`) fused by **Reciprocal Rank Fusion (RRF)**.  
- **Optional reranking**: **CrossEncoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for local CPU rerank (if installed).  
- **Low-recall rescue**: simple query reformulation heuristic tries paraphrases and picks the variant with more unique sources.

### Command Line Interface
- **Input handling**: `typer` CLI → `python main.py --q "your question"`.  
- **Prompting**: strict system prompt (use only context; say “I don't know…” if missing).  
- **Output**: final answer + **source list (filename + page)**.  
- **Error handling**: graceful guardrail denial message; robust ingest warnings.

### Guardrails
- **Keyword-based denylist** (`safety.py`), configurable topics and message in `config.yml`.  
- **Fail-open** behavior outside denylist (no over-blocking benign queries).  

### Robustness & Efficiency
- **Local-only**: no network calls.  
- **Performance knobs**:
  - Threads (`n_threads`), context (`n_ctx`), temperature, and max tokens in `config.yml`.  
  - Embedding device can be CPU/GPU.  
- **Resource minded**: Q4 model + CPU threads tuned for typical 16GB RAM laptops.  
- **Config-driven**: paths, LLM, retrieval, and CLI toggles centralized in `config.yml`.

---

## 2) Repository Layout

```
.
├─ corpus/
├─ index/
├─ models/
├─ ingest.py
├─ main.py
├─ retriever.py
├─ safety.py
├─ config.yml
├─ requirements.txt
└─ eval.py
```

---

## 3) Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build the index

```bash
python ingest.py
```

### Ask a question

```bash
python main.py --q "give the summary of the data"
```

---

## 4) Configuration (config.yml)

Example:

```yaml
paths:
  pdf_dir: "./corpus"
  index_dir: "./index"
  gguf_path: "./models/Phi-3-mini-4k-instruct-q4.gguf"

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

chunking:
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  strategy: "hybrid"
  top_k: 8
  bm25_weight: 0.5
  mmr: false
  low_recall_threshold: 2

llm:
  n_ctx: 4096
  n_threads: 6
  n_gpu_layers: 0
  temperature: 0.2
  max_new_tokens: 320

guardrails:
  enabled: true
  blocked_topics: ["self-harm", "explicit_illegal_howto", "malware", "explosives", "hate"]
  deny_message: "I can’t help with that topic. If you have another question, I’m happy to help with safe, allowed topics."
```

---

## 5) License

Local research/demo use. Check individual model/data licenses before redistribution.
