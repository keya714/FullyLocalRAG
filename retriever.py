from pathlib import Path
import yaml
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CONFIG_PATH = "config.yml"

def _cfg(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _emb(name, device):
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_retriever(cfg_path: str = CONFIG_PATH) -> FAISS:
    cfg = _cfg(cfg_path)
    vs = FAISS.load_local(
        cfg["paths"]["index_dir"],
        _emb(cfg["embedding"]["model_name"], cfg["embedding"]["device"]),
        allow_dangerous_deserialization=True
    )
    return vs

def retrieve(vs: FAISS, query: str, k: int = 5, strategy: str = "similarity"):
    if strategy == "mmr":
        return vs.max_marginal_relevance_search(query, k=k, fetch_k=max(20, 3*k), lambda_mult=0.5)
    elif strategy == "similarity_with_scores":
        return vs.similarity_search_with_score(query, k=k)
    else:
        return vs.similarity_search(query, k=k)

def format_context(docs) -> str:
    """Return a context string suitable for prompting an LLM (with inline tags)."""
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        tag = f"{Path(src).name}#p{page}"
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"[{tag}] {snippet}")
    return "\n\n".join(lines)
