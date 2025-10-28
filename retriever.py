from __future__ import annotations
import yaml
from pathlib import Path
from typing import List
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


CONFIG_PATH = "config.yml"

def load_cfg(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_vector_store(cfg_path: str = CONFIG_PATH) -> FAISS:
    cfg = load_cfg(cfg_path)
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["embedding"]["model_name"],
        model_kwargs={"device": cfg["embedding"]["device"]},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(
        cfg["paths"]["index_dir"],
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vs

@dataclass
class RetrievalConfig:
    top_k: int
    strategy: str
    bm25_weight: float
    mmr: bool

def retrieve(vs: FAISS, query: str, k: int, strategy: str, bm25_weight: float, mmr: bool) -> List[Document]:
    # For now: simple dense retrieval
    if strategy == "dense":
        docs = vs.similarity_search(query=query, k=k)
    else:
        # fallback to dense if others not implemented
        docs = vs.similarity_search(query=query, k=k)
    return docs

def format_context(docs: List[Document]) -> str:
    lines: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        tag = f"{Path(src).name}#p{page}"
        snippet = (d.page_content or "").strip().replace("\n", " ")
        lines.append(f"[{tag}] {snippet}")
    return "\n\n".join(lines)

def unique_source_count(docs: List[Document]) -> int:
    return len({d.metadata.get("source") for d in docs})
