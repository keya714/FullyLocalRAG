#!/usr/bin/env bash

echo "==============================================="
echo "  Setting up Fully Local RAG environment..."
echo "==============================================="

# ---------------------------
# 1. Check Python
# ---------------------------
if ! command -v python &> /dev/null; then
    echo " Python not found. Please install Python 3.10+ first."
    exit 1
fi

PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo " Python version: $PYVER"

# ---------------------------
# 2. Create virtual environment
# ---------------------------
echo " Creating virtual environment (.venv)"
python3 -m venv .venv
source .venv/bin/activate

# ---------------------------
# 3. Upgrade pip + install deps
# ---------------------------
echo "  Installing dependencies..."
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core RAG deps
pip install -r requirements.txt

# ---------------------------
# 4. Prepare folder structure
# ---------------------------
mkdir -p corpus index models

if [ ! -f config.yml ]; then
    echo "🧾 Creating default config.yml..."
    cat > config.yml <<'YAML'
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

cli:
  show_sources: true
YAML
fi

# ---------------------------
# 5. Download sample model (optional)
# ---------------------------
if [ ! -f ./models/Phi-3-mini-4k-instruct-q4.gguf ]; then
    echo "⬇  Model not found. Downloading sample Phi-3-mini Q4 GGUF (~1.9GB)..."
    wget -O ./models/Phi-3-mini-4k-instruct-q4.gguf \
      https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf \
      || echo "  Model download failed. Please place your GGUF model manually in ./models/"
else
    echo " Model already exists in ./models/"
fi

# ---------------------------
# 6. Build FAISS index (if PDFs present)
# ---------------------------
if [ -d corpus ] && [ "$(ls -A corpus 2>/dev/null)" ]; then
    echo " PDFs found in corpus/. Running ingest.py..."
    python ingest.py
else
    echo "  No PDFs found in ./corpus. Skipping ingestion."
    echo "   → Add PDFs and run: python ingest.py"
fi

# ---------------------------
# 7. Test query
# ---------------------------
echo " Testing pipeline..."
python main.py --q "who are you" || true

echo "Setup complete!"
echo "-----------------------------------------------"
echo "Activate environment next time with:"
echo "  source .venv/bin/activate"
echo "Ask questions using:"
echo "  python main.py --q 'your question here'"
echo "-----------------------------------------------"
