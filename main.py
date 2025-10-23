#!/usr/bin/env python3
"""
Full RAG query pipeline (HF Transformers version):
- Loads FAISS index and embeddings from config.yml
- Retrieves top-k chunks
- Builds a context-aware chat prompt via tokenizer.apply_chat_template
- Calls a lightweight HF chat model (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Prints the final grounded answer with citations
"""

import yaml
from pathlib import Path
from retriever import load_retriever, retrieve, format_context  # keep your existing retriever module
from transformers import AutoTokenizer, pipeline

# ---------------------------------------------------------------------
# System instruction (goes in the system role)
# ---------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a helpful and precise assistant. Use ONLY the information provided "
    "in the context to answer the question. If the answer cannot be found in the "
    "context, reply: \"I don't know based on the provided documents.\" "
    "Cite evidence inline like [source#page]. Never invent information."
)

def load_cfg(path: str = "config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_messages(question: str, context: str):
    """Chat-style messages for models with a chat template."""
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"}
    ]

def run_rag(query: str, cfg_path: str = "config.yml", k: int = 5):
    """End-to-end RAG with HF Transformers chat model."""
    cfg = load_cfg(cfg_path)

    # ---------------- 1) Retrieve ----------------
    vs = load_retriever(cfg_path)
    docs = retrieve(vs, query, k=k)
    context = format_context(docs)  # produces [file#pN] tagged snippets

    # ---------------- 2) Build LLM ----------------
    # Read model settings from config if present, else sensible defaults.
    llm_cfg = cfg.get("llm", {})
    model_id = llm_cfg.get("hf_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    temperature = float(llm_cfg.get("temperature", 0.2))
    max_new_tokens = int(llm_cfg.get("max_new_tokens", 300))

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use the model's chat template to create a proper prompt string
    messages = build_messages(query, context)
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   # append the assistant header
        tokenize=False                # produce plain text for pipeline()
    )

    # Create a generation pipeline (CPU/GPU auto)
    gen = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
        return_full_text=False  # only return the assistant continuation
    )

    # ---------------- 3) Generate ----------------
    out = gen(prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
    answer = out[0]["generated_text"].strip()

    # ---------------- 4) Display ----------------
    print("\n" + "="*80)
    print(f"QUESTION: {query}")
    print("="*80)
    print(answer)
    print("\n---\nSources:")
    for d in docs:
        print(f"- {Path(d.metadata.get('source')).name} (page {d.metadata.get('page', 'n/a')})")
    print("="*80)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ask a question to your local RAG pipeline (HF Transformers).")
    ap.add_argument("--q", "--query", default="Which area of india is considered for LULC", dest="query", help="User question")
    ap.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    ap.add_argument("--config", type=str, default="config.yml", help="Path to config.yml")
    args = ap.parse_args()

    run_rag(args.query, cfg_path=args.config, k=args.k)
