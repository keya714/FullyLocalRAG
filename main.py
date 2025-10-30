#!/usr/bin/env python3
from __future__ import annotations
import yaml
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import os

import typer
from rich.console import Console

from retriever import load_vector_store, retrieve, format_context, unique_source_count
from models import OllamaConfig, OllamaWrapper

from safety import GuardConfig, check_guardrails
import torch

torch.set_num_threads(max(1, os.cpu_count() // 2))
app = typer.Typer(add_completion=False)
console = Console()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide INFO and WARNING logs from TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimization message

# ----------------------------
# System prompts
# ----------------------------
SYSTEM_MSG = (
    "You are a precise and factual assistant. "
    "Use ONLY the information explicitly provided in the context below to answer the user's question. "
    "If the answer is not clearly stated in the context, reply exactly with: "
    "\"I don't know based on the provided documents.\" "
    "Do not use prior knowledge or make assumptions. "
    "Answer concisely in complete sentences."
)

MAP_SYSTEM = (
    "You extract only facts that directly answer the user's question. "
    "If a chunk does not contain answerable facts, output exactly: NO-FACT."
)

# ----------------------------
# Helpers
# ----------------------------
def load_cfg(path: str = "config.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def build_llama_prompt(messages: list[dict[str, str]]) -> str:
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append(f"<|system|>\n{m['content']}\n")
        elif m["role"] == "user":
            out.append(f"<|user|>\n{m['content']}\n")
        elif m["role"] == "assistant":
            out.append(f"<|assistant|>\n{m['content']}\n")
    out.append("<|assistant|>\n")
    return "\n".join(out)

def maybe_reformulate(query: str) -> List[str]:
    q = query.strip()
    alts = [q]
    if len(q.split()) <= 3:
        alts.append(f"Provide detailed information about: {q}")
        alts.append(f"What are the key facts and definitions related to {q}?")
    else:
        alts.append(" ".join([w.strip("?,.") for w in q.split() if len(w) > 3]))
    return alts

def is_reference_chunk(d) -> bool:
    t = (d.page_content or "").lower()
    return ("references" in t or "bibliography" in t) and len(t) < 1200

# ----------------------------
# Map → Reduce prompts
# ----------------------------
def summarize_chunk(llm: OllamaWrapper, question: str, tag: str, text: str) -> Optional[str]:
    """
    Returns bullet lines prefixed with (tag) for relevant facts, or None if NO-FACT.
    """
    # Reset KV cache to avoid bleed across chunks
    try:
        llm.llm.reset()  # llama.cpp API; safe to try
    except Exception:
        pass

    msgs = [
        {"role":"system","content": MAP_SYSTEM},
        {"role":"user","content": (
            f"Question: {question}\n\n"
            f"Chunk [{tag}]:\n{text}\n\n"
            f"Instructions:\n"
            f"- If relevant facts exist, list 1–3 bullet points with exact terms/numbers.\n"
            f"- Else, output exactly: NO-FACT"
        )}
    ]
    prompt = build_llama_prompt(msgs)
    out = llm.chat(prompt).strip()

    if out.startswith("NO-FACT"):
        return None

    # Normalize to bullets with citation tags
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        ln = ln.lstrip("-• ").strip()
        if ln:
            bullets.append(f"- ({tag}) {ln}")
    return "\n".join(bullets) if bullets else None

def answer_from_digest(llm: OllamaWrapper, question: str, digest: str) -> str:
    # Reset KV cache before final reduce
    try:
        llm.llm.reset()
    except Exception:
        pass

    msgs = [
        {"role":"system","content": SYSTEM_MSG},
        {"role":"user","content": f"Question: {question}\n\nBullet digest:\n{digest}"}
    ]
    prompt = build_llama_prompt(msgs)
    return llm.chat(prompt).strip()

# ----------------------------
# CLI command
# ----------------------------
@app.command()
def chat(
    q: str = typer.Option(..., "--q", "--query", help="User query"),
    k: int = typer.Option(None, "--k", help="Top-k chunks to retrieve (override config)"),
    config: str = typer.Option("config.yml", "--config", help="Path to config file"),
):
    t0 = time.perf_counter()
    cfg = load_cfg(config)

    # Guardrails
    gcfg = GuardConfig(
        enabled=bool(cfg.get("guardrails", {}).get("enabled", True)),
        blocked_topics=cfg.get("guardrails", {}).get("blocked_topics"),
        deny_message=cfg.get("guardrails", {}).get("deny_message")
    )
    blocked, msg = check_guardrails(q, gcfg)
    if blocked:
        retrieval_time = time.perf_counter() - t0
        console.print(f"[red]{msg}[/]")
        console.print(f"[cyan]LLM generation took:[/cyan] [bold]{retrieval_time:.2f}s[/bold]")
        console.print(f"[yellow]Total time:[/yellow] [bold]{retrieval_time:.2f}s[/bold]")
        sys.exit(0)

    # Retrieval config
    strategy = cfg["retrieval"].get("strategy", "hybrid")
    top_k = k or int(cfg["retrieval"].get("top_k", 5))
    bm25_weight = float(cfg["retrieval"].get("bm25_weight", 0.5))
    mmr = bool(cfg["retrieval"].get("mmr", False))

    # Load index
    vs = load_vector_store(config)

    # Retrieve (with your hybrid defaults)
    docs = retrieve(vs, q, k=8, strategy="hybrid", bm25_weight=0.5, mmr=True)

    # Low-recall rescue by reformulation
    min_sources = int(cfg["retrieval"].get("low_recall_threshold", 2))
    if unique_source_count(docs) < min_sources:
        variants = maybe_reformulate(q)
        best_docs = None
        best_count = 0
        for v in variants:
            cand_docs = retrieve(vs, v, k=top_k, strategy=strategy, bm25_weight=bm25_weight, mmr=mmr)
            count = unique_source_count(cand_docs)
            if count > best_count:
                best_count = count
                best_docs = cand_docs
        if best_docs is not None:
            docs = best_docs

    # Optional: drop reference-like chunks
    docs = [d for d in docs if not is_reference_chunk(d)]

    # For debugging: show raw context
    context = format_context(docs)
    console.print("[bold]Chunks used for context:[/bold]")
    print(context)
    print("==="*10)
    sys.stdout.flush()
    t1 = time.perf_counter()

    # Load LLM (Ollama)
    lcfg = OllamaConfig(
        model=str(cfg["llm"].get("model", "llama3.2")),
        num_ctx=int(cfg["llm"].get("n_ctx", 4096)),
        temperature=float(cfg["llm"].get("temperature", 0.2)),
        max_new_tokens=int(cfg["llm"].get("max_new_tokens", 320)),
        stop=tuple(cfg["llm"].get("stop", [])),
    )
    llm = OllamaWrapper(lcfg)

    # ----------------------------
    # MAP: per-chunk relevance summaries
    # ----------------------------
    t_map0 = time.perf_counter()
    summaries: List[str] = []
    for d in docs:
        src = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page", "n/a")
        tag = f"{src}#p{page}"
        chunk_text = (d.page_content or "").strip()
        s = summarize_chunk(llm, q, tag, chunk_text)
        if s:
            summaries.append(chunk_text)
    t_map1 = time.perf_counter()

    # If nothing relevant, immediate fallback
    if not summaries:
        console.print("\n" + "="*80)
        console.print(f"[bold]QUESTION:[/bold] {q}")
        console.print("="*80)
        console.print("I don't know based on the provided documents.")

        if bool(cfg.get("cli", {}).get("show_sources", True)):
            console.print("\n---\n[bold]Sources:[/bold]")
            for d in docs:
                src = Path(d.metadata.get("source", "")).name
                page = d.metadata.get("page", "n/a")
                console.print(f"- {src} (page {page})")
        console.print("="*80)

        retrieval_time = t1 - t0
        map_time = t_map1 - t_map0
        generation_time = 0.0
        total_time = (time.perf_counter() - t0)

        console.print(f"[cyan]Retrieval took:[/cyan] [bold]{retrieval_time:.2f}s[/bold]")
        console.print(f"[magenta]Map step took:[/magenta] [bold]{map_time:.2f}s[/bold]")
        console.print(f"[cyan]LLM generation took:[/cyan] [bold]{generation_time:.2f}s[/bold]")
        console.print(f"[yellow]Total time:[/yellow] [bold]{total_time:.2f}s[/bold]")
        return

    # Keep digest small to fit context nicely
    MAX_BULLETS = 12
    digest = "\n".join(summaries[:MAX_BULLETS])

    # ----------------------------
    # REDUCE: final answer from digest
    # ----------------------------
    t_red0 = time.perf_counter()
    final_answer = answer_from_digest(llm, q, digest)
    t_red1 = time.perf_counter()

    # Print final
    console.print("\n" + "="*80)
    console.print(f"[bold]QUESTION:[/bold] {q}")
    console.print("="*80)
    console.print(final_answer)

    if bool(cfg.get("cli", {}).get("show_sources", True)):
        console.print("\n---\n[bold]Sources:[/bold]")
        for d in docs:
            src = Path(d.metadata.get("source", "")).name
            page = d.metadata.get("page", "n/a")
            console.print(f"- {src} (page {page})")
    console.print("="*80)

    # Timings
    retrieval_time = t1 - t0
    map_time = t_map1 - t_map0
    generation_time = (t_red1 - t_red0)
    total_time = (time.perf_counter() - t0)

    console.print(f"[cyan]Retrieval took:[/cyan] [bold]{retrieval_time:.2f}s[/bold]")
    console.print(f"[magenta]Map step took:[/magenta] [bold]{map_time:.2f}s[/bold]")
    console.print(f"[cyan]LLM generation took:[/cyan] [bold]{generation_time:.2f}s[/bold]")
    console.print(f"[yellow]Total time:[/yellow] [bold]{total_time:.2f}s[/bold]")

if __name__ == "__main__":
    app()
