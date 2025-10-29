from __future__ import annotations
import re
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Optional (small & CPU-friendly) cross-encoder for local re-ranking
try:
    from sentence_transformers import CrossEncoder  # pip install sentence-transformers
    HAS_CROSS = True
except Exception:
    HAS_CROSS = False

# Lightweight BM25 (pure Python, CPU)
try:
    from rank_bm25 import BM25Okapi  # pip install rank_bm25
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

CONFIG_PATH = "config.yml"


# -------------------- Config --------------------
def load_cfg(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------- Vector store --------------------
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


# -------------------- In-memory BM25 built from FAISS docs --------------------
def _simple_tokens(text: str) -> List[str]:
    # Simple, fast tokenizer (no external deps)
    return [t for t in re.split(r"\W+", text.lower()) if t]

def build_bm25_from_faiss(vs: FAISS) -> Tuple[Optional[BM25Okapi], List[str]]:
    """
    Builds a BM25 index over the FAISS docstore contents for hybrid retrieval.
    Returns (bm25, ids) where ids[i] maps back to FAISS docstore.
    """
    if not HAS_BM25:
        return None, []
    # langchain FAISS keeps docs in vs.docstore._dict with keys=ids
    store_dict: Dict[str, Document] = getattr(vs.docstore, "_dict", {})
    ids: List[str] = list(store_dict.keys())
    corpus_tokens = [_simple_tokens(store_dict[i].page_content or "") for i in ids]
    bm25 = BM25Okapi(corpus_tokens) if ids else None
    return bm25, ids


# -------------------- Utilities --------------------
def rrf(ranked_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion over ranked indices (not ids).
    Returns mapping: local_index -> fused_score
    """
    from collections import defaultdict
    scores = defaultdict(float)
    for rl in ranked_lists:
        for rank, local_idx in enumerate(rl, start=1):
            scores[local_idx] += 1.0 / (k + rank)
    return scores

def _topk_by_score(items: List[Tuple[int, float]], k: int) -> List[int]:
    return [i for (i, _) in sorted(items, key=lambda x: x[1], reverse=True)[:k]]


@dataclass
class RetrievalConfig:
    top_k: int = 8
    strategy: str = "hybrid"   # "dense" | "bm25" | "hybrid"
    bm25_weight: float = 0.5   # not used directly; RRF is weight-agnostic
    mmr: bool = False          # use FAISS max-marginal-relevance
    top_k_rerank: int = 8      # cross-encoder output size
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small CPU model
    use_cross_encoder: bool = True


# -------------------- Retrieval core --------------------
class HybridRetriever:
    def __init__(self, vs: FAISS, cfg: RetrievalConfig):
        self.vs = vs
        self.cfg = cfg
        self._bm25, self._bm25_ids = build_bm25_from_faiss(vs)
        self._cross: Optional[CrossEncoder] = None
        if self.cfg.use_cross_encoder and HAS_CROSS:
            try:
                self._cross = CrossEncoder(self.cfg.cross_encoder_model)
            except Exception:
                self._cross = None

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (local_index, score) against vs.docstore ids order.
        We use similarity_search_with_score to get FAISS scores directly.
        """
        docs_and_scores = self.vs.similarity_search_with_score(query=query, k=max(k, 50))
        # Map back to local indices in docstore order for fusion
        store_dict: Dict[str, Document] = getattr(self.vs.docstore, "_dict", {})
        id_to_local = {doc_id: idx for idx, doc_id in enumerate(store_dict.keys())}
        pairs: List[Tuple[int, float]] = []
        for d, score in docs_and_scores:
            # FAISS returns lower distance = better; invert into similarity
            sim = -float(score)
            # Find local index by doc_id (LangChain stores it in d.metadata.get("doc_id") sometimes,
            # but we can search by identity in docstore)
            # We attempt to get the underlying id via d.metadata["_id"] if present; else find first match.
            doc_id = d.metadata.get("doc_id") or d.metadata.get("_id")
            local_idx = None
            if doc_id and doc_id in id_to_local:
                local_idx = id_to_local[doc_id]
            else:
                # fallback: linear probe by text equality (rare)
                for _id, idx in id_to_local.items():
                    if store_dict[_id].page_content == d.page_content:
                        local_idx = idx
                        break
            if local_idx is not None:
                pairs.append((local_idx, sim))
        return pairs

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not (HAS_BM25 and self._bm25 and self._bm25_ids):
            return []
        tokens = _simple_tokens(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        return ranked[:max(k, 50)]

    def _fetch_docs(self, local_indices: List[int]) -> List[Document]:
        store_dict: Dict[str, Document] = getattr(self.vs.docstore, "_dict", {})
        ids = list(store_dict.keys())
        out: List[Document] = []
        for i in local_indices:
            if 0 <= i < len(ids):
                out.append(store_dict[ids[i]])
        return out

    def _mmr_reduce(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        # Use built-in MMR over FAISS by passing vectors through the vectorstore again.
        # Here we just call max_marginal_relevance_search which re-retrieves,
        # but with small k it’s OK for quality.
        try:
            return self.vs.max_marginal_relevance_search(query=query, k=k, fetch_k=min(20, max(k*3, 20)))
        except Exception:
            return candidates[:k]

    def _cross_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        if not (self._cross and docs):
            return docs[:k]
        pairs = [(query, d.page_content) for d in docs]
        scores = self._cross.predict(pairs)  # higher is better
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in ranked[:k]]

    def retrieve(self, query: str) -> List[Document]:
        k = max(1, self.cfg.top_k)

        if self.cfg.strategy == "dense":
            dense = self._dense_search(query, k=k)
            local_idxs = [i for i, _ in dense[:k]]
        elif self.cfg.strategy == "bm25":
            bm25 = self._bm25_search(query, k=k)
            local_idxs = [i for i, _ in bm25[:k]]
        else:
            # HYBRID with RRF
            dense = self._dense_search(query, k=k)
            bm25 = self._bm25_search(query, k=k)
            # Convert lists into ranked index lists (just order by score desc)
            dense_ranked = [i for i, _ in sorted(dense, key=lambda x: x[1], reverse=True)]
            bm25_ranked = [i for i, _ in sorted(bm25, key=lambda x: x[1], reverse=True)]
            fused_scores = rrf([dense_ranked, bm25_ranked], k=60)
            fused = _topk_by_score(list(fused_scores.items()), k=max(k, 12))
            local_idxs = fused[:max(k, 12)]

        # Fetch documents
        candidates = self._fetch_docs(local_idxs)

        # Optional MMR diversity
        if self.cfg.mmr and candidates:
            candidates = self._mmr_reduce(query, candidates, k=max(k, len(candidates)))

        # Optional cross-encoder re-rank (final top_k_rerank)
        final = self._cross_rerank(query, candidates, k=min(self.cfg.top_k_rerank, k))

        # Ensure exactly top_k docs
        return final[:k]


# -------------------- Public API (backwards-compatible) --------------------
def retrieve(vs: FAISS, query: str, k: int, strategy: str, bm25_weight: float, mmr: bool) -> List[Document]:
    cfg = RetrievalConfig(top_k=k, strategy=strategy, bm25_weight=bm25_weight, mmr=mmr)
    retr = HybridRetriever(vs, cfg)
    return retr.retrieve(query)


def format_context(docs: List[Document]) -> str:
    lines: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        heading = d.metadata.get("heading") or d.metadata.get("title") or ""
        tag = f"{Path(src).name}#p{page}"
        snippet = (d.page_content or "").strip().replace("\n", " ")
        heading_part = f" — {heading}" if heading else ""
        lines.append(f"[{tag}{heading_part}] {snippet}")
    return "\n\n".join(lines)

def unique_source_count(docs: List[Document]) -> int:
    return len({d.metadata.get("source") for d in docs})
