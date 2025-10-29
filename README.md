# Fully-Local RAG (CLI) — Laptop-Ready

A Retrieval-Augmented Generation (RAG) system that runs **entirely on a laptop**, with no cloud calls.

- **LLM:** Local Ollama setup running Llama 3.2 as the large language model, which is already quantized.
- **Retrieval:** Hybrid retrieval combining Dense FAISS and BM25, with optional cross-encoder reranking.
- **CLI:** Allows users to ask questions from the terminal; outputs the answer along with sources and response time.
- **Guardrails:** Implements a fast keyword deny-list, configurable for safety.
- **Query Checking:** Uses an LLM-based intent recognition mechanism to verify if a query is relevant to the fetched chunks before answering.


## 1) Features mapped to the Challenge Requirements

### Language Model
| Feature            | Description                                                                                                   |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| Pre-trained Model    | Uses Ollama's **Llama 3.2**, a pre-trained, instruction-tuned model available in quantized formats.           |
| Quantization        | Employs advanced quantization techniques (4-bit groupwise for weights, 8-bit dynamic for activations) for fast and memory-efficient inference. |
| Local Inference      | Runs efficiently on CPU-only systems with configurable threading; supports optional GPU acceleration via Ollama. |
| Suitability          | Designed for resource-constrained environments like laptops, with quantization-aware training and LoRA fine-tuning for robustness.               |


### Retrieval Mechanism

| Feature | Description |
|---------|--------------|
| **Corpus Creation** | Ingests PDFs from `./corpus/` (configurable). |
| **Chunking** | Uses `RecursiveCharacterTextSplitter` with overlap, including basic cleaning to reduce noise. |
| **Embeddings** | Utilizes Sentence Transformers from HuggingFace, with device configuration options (CPU/GPU). |
| **Indexing** | Stores the FAISS index on disk in `./index`. |
| **Hybrid Retrieval** | Combines dense FAISS retrieval and BM25 (via `rank_bm25`) with results fused by **Reciprocal Rank Fusion (RRF)**. |
| **Optional Reranking** | Employs a **CrossEncoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for local CPU reranking if installed. |
| **Low-recall Rescue** | Implements a heuristic for query reformulation, paraphrasing, and selecting the variant with more unique sources. |

### Different Retrival Comparison

| Aspect            | With Cross-Encoder                    | Without Cross-Encoder                 |
|-------------------|------------------------------------|-------------------------------------|
| **Accuracy**      | Higher precision; understands query-doc interactions deeply | Lower precision; relies on separate embeddings only |
| **Speed**         | Slower; computes scores for each query-doc pair | Faster; uses approximate similarity scores only |
| **Workflow**      | Two-step: retrieve candidates → rerank top results | Single-step retrieval and ranking |
| **Result Quality**| More relevant and refined final results | Less refined rankings, may return less relevant docs |
| **Best For**      | Use cases needing high accuracy and precision | Use cases prioritizing speed and scalability |



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
├─ corpus/   # put PDFs here (example: LULC_Paper.pdf)
├─ index/                     # built FAISS index + manifest
│  ├─ index.faiss
│  ├─ index.pkl
│  └─ manifest.txt
├─ models.py
├─ ingest.py                  # build index from PDFs
├─ main.py                    # CLI app (ask questions)
├─ retriever.py               # hybrid retrieval + RRF (+ optional reranker)
├─ safety.py                  # keyword guardrails
├─ config.yml                 # all configuration
├─ requirements.txt
```

---

## 3) Quickstart
###Prereqs

Python 3.10+ recommended
On CPU-only laptops: works out of the box.
(Optional) If using GPU for embeddings or llama.cpp offload, ensure CUDA toolchain is set up.

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

### Ollama Setup

1. **Install Ollama**  
Download and install Ollama from the official site:  
[https://ollama.com/download](https://ollama.com/download)

2. **Pull the Llama 3.2 Model**  
Fetch the Llama 3.2 model via Ollama CLI: ollama pull llama3.2

3. **Start Ollama Service**  
Ensure the Ollama local service is running:  ollama serve

### Ask a question

```bash
python main.py --q "give the summary of the data"
```

---

## 4) Configuration (config.yml)

Example:

```yaml
paths:
  pdf_dir: ./corpus
  index_dir: ./index

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu

chunking:
  chunk_size: 900
  chunk_overlap: 120

retrieval:
  use_cross_encoder: true
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_rerank: 8
  top_k: 5
  strategy: hybrid        # dense | bm25 | hybrid (works if you used my hybrid retriever)
  bm25_weight: 0.5
  mmr: false
  min_context_chars: 800
  low_recall_threshold: 2

llm:
  model: "llama3.2"   # or any model you `ollama pull`
  n_ctx: 4096
  temperature: 0.3
  max_new_tokens: 512
  stop: []               # e.g., ["<|assistant|>", "</s>"]


guardrails:
  enabled: true
  blocked_topics:
    - self-harm
    - explicit_illegal_howto
    - malware
    - explosives
    - hate
  deny_message: >
    I can’t help with that topic. If you have another question, I’m happy to help with safe, allowed topics.

cli:
  pretty_print: true
  show_sources: true
  max_source_chars: 1000

```

---

### 5) How it Works (Pipeline)
* Ollama Setup
  Download any llm model(here llama3.2) and run ollama(command is ollama serve)


- **Ingest (ingest.py)**  
- Load PDFs page-by-page with source and page metadata using PyPDFLoader.  
- Clean text by removing noisy symbols and collapsing whitespace.  
- Chunk documents with overlap for context continuity.  
- Embed chunks using sentence-transformers and build a FAISS index, saved to `./index`.

- **Guardrails**  
- Check if the query is safe and allowed.  
- If blocked, generate a static denial response.

- **Retrieve (retriever.py)**  
- Load FAISS index and embeddings from disk.  
- Perform hybrid retrieval: dense similarity search via FAISS and sparse retrieval with BM25.  
- Fuse results using Reciprocal Rank Fusion (RRF).  
- Optionally rerank top results with CrossEncoder (if installed).

- **Generate (main.py)**  
- Perform guardrail checks; deny unsafe queries.  
- If retrieval yield is low, reformulate query variants and pick the best.  
- Filter out low-value reference chunks.  
- Format context and send for LLM generation.  
- Use strict prompt instructing the LLM to answer only from the context or say "I don't know".  
- Print concise answer with source list and timing info.

---

### 6) Improving RAG Pipeline

This system uses a two-stage reasoning (Map-Reduce) to ensure answers come strictly from retrieved documents and reduce hallucinations.

#### How it works:

1. **Retrieve**  
 The hybrid retriever fetches top-k relevant chunks from the local corpus.

2. **Map Step – Per-Chunk Summarization**  
 Each chunk is independently sent to the LLM with strict instructions:  
 - Extract only direct facts answering the question.  
 - If none exist, output "NO-FACT".  
 - Return 1-3 factual bullet points tagged with source citations.

3. **Reduce Step – Final Decision**  
 The LLM receives a compressed digest of all factual bullets. It then:  
 - Generates a concise, citation-based final answer using only those facts, or  
 - Replies explicitly with "I don't know based on the provided documents." if insufficient information is available.

**Output:**  
Short, verifiable answers grounded in local data with automatic fallback if context is lacking.

---

This design ensures reliable, traceable answers based on your local knowledge base powered by Ollama’s Llama 3.2, combined with robust retrieval and safety mechanisms.


### For Best Results

- **Adjust Chunk Size According to Document Length:**  
  Tailor chunk sizes based on the length and type of documents you ingest.  
  - Smaller chunks (128–256 tokens) work well for fact-based queries needing precision.  
  - Larger chunks (256–512 tokens) are better for tasks requiring broader context, like summaries.  
  Experimentation and aligning chunk size with your model’s context window are key.

- **Tune Temperature Settings:**  
  Adjust the LLM temperature based on the type of information desired:  
  - Lower temperatures (e.g., 0.1–0.3) produce more focused, deterministic answers.  
  - Higher temperatures (e.g., 0.7+) encourage creativity or exploratory responses.

- **Frequent Cache Clearing:**  
  Regularly delete temporary cache folders (such as `__pycache__`) and any model inference caches to:  
  - Avoid stale or conflicting context during repeated queries.  
  - Ensure cleaner, more accurate generation by preventing cache pollution.

These practices help enhance retrieval accuracy, LLM response quality, and overall system stability.

### Performance Metrics Display

After each query response, the system displays key timing metrics to provide quantitative insight into performance:

| Metric              | Description                                         |
|---------------------|-----------------------------------------------------|
| **Retrieval Time**  | Time taken to fetch relevant documents from the index, reflecting retrieval speed and efficiency. |
| **Response Time**   | Time taken by the language model to generate an answer based on the retrieved context.              |
| **Total Time**      | Combined end-to-end time from query initiation to final answer display, illustrating overall latency. |

These metrics help showcase the system's responsiveness and resource usage on typical hardware, aligning with the challenge’s emphasis on robust and efficient local operation.


### Flowchart:

A[User Query (CLI Input)] --> B[Guardrails Check]
B -->|Unsafe Query| B1[Deny Message<br>(from config.yml)]
B -->|Safe Query| C[Retrieve Relevant Chunks]

subgraph Retrieval
    C --> D1[Dense Retrieval (FAISS)]
    C --> D2[Sparse Retrieval (BM25)]
    D1 --> D3[RRF Fusion]
    D2 --> D3
    D3 -->|Optional| D4[Cross-Encoder Reranking]
end

D4 --> E[Selected Top-k Chunks]
D3 --> E

E --> F[Query-Context Validation<br>(LLM Intent Check)]
F -->|Irrelevant| F1[Query Reformulation + Retry]
F -->|Relevant| G[LLM Generation (Ollama)]

subgraph Generation
    G1[Map Step:<br>Per-chunk factual extraction]
    G2[Reduce Step:<br>Aggregate & compose final answer]
end

E --> G1 --> G2 --> H[Final Answer]

H --> I[Output to CLI<br>Answer + Sources + Timings]

%% Supporting Modules
subgraph Ingest Pipeline
    I1[Load PDFs (PyPDFLoader)]
    I2[Chunking & Cleaning]
    I3[Embeddings (Sentence Transformer)]
    I4[Index Build (FAISS + BM25)]
    I1 --> I2 --> I3 --> I4
end

%% Relations
I4 -.-> C
config[config.yml: settings & paths] -.-> B
config -.-> C
config -.-> G

%% Metrics
H --> M[Metrics Display:<br>Retrieval Time / LLM Time / Total Time]
