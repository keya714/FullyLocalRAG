#!/usr/bin/env python3
from __future__ import annotations
import yaml
import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import typer
from rich.console import Console

from retriever import load_vector_store, retrieve, format_context, unique_source_count
from models import HFConfig, LlamaCppConfig, LlamaCppWrapper, load_transformers_chat, apply_chat_template
from safety import GuardConfig, check_guardrails

app = typer.Typer(add_completion=False)
console = Console()

SYSTEM_MSG = (
    "You are a helpful and precise assistant. Use ONLY the information provided "
    "in the context to answer the question. If the answer cannot be found in the "
    "context, reply: \"I don't know based on the provided documents.\" "
    "Cite evidence inline like [source#page]. Never invent information."
)

def load_cfg(path: str = "config.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_messages(question: str, context: str) -> List[Dict[str,str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"}
    ]

def maybe_reformulate(query: str) -> List[str]:
    q = query.strip()
    alts = [q]
    if len(q.split()) <= 3:
        alts.append(f"Provide detailed information about: {q}")
        alts.append(f"What are the key facts and definitions related to {q}?")
    else:
        alts.append(" ".join([w.strip("?,.") for w in q.split() if len(w) > 3]))
    return alts

@app.command()
def chat(
    q: str = typer.Option(..., "--q", "--query", help="User query"),
    k: int = typer.Option(None, "--k", help="Top-k chunks to retrieve (override config)"),
    config: str = typer.Option("config.yml", "--config", help="Path to config file"),
):
    cfg = load_cfg(config)

    gcfg = GuardConfig(
        enabled=bool(cfg.get("guardrails", {}).get("enabled", True)),
        blocked_topics=cfg.get("guardrails", {}).get("blocked_topics"),
        deny_message=cfg.get("guardrails", {}).get("deny_message")
    )
    blocked, msg = check_guardrails(q, gcfg)
    if blocked:
        console.print(f"[red]{msg}[/]")
        sys.exit(0)

    strategy = cfg["retrieval"].get("strategy", "dense")
    top_k = k or int(cfg["retrieval"].get("top_k", 5))
    bm25_weight = float(cfg["retrieval"].get("bm25_weight", 0.5))
    mmr = bool(cfg["retrieval"].get("mmr", False))

    vs = load_vector_store(config)

    docs = retrieve(vs, q, k=top_k, strategy=strategy, bm25_weight=bm25_weight, mmr=mmr)

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

    context = format_context(docs)
    print(context)
    # backend = cfg["llm"].get("backend", "transformers")
    temperature = float(cfg["llm"].get("temperature", 0.2))
    max_new_tokens = int(cfg["llm"].get("max_new_tokens", 320))

    lcfg = LlamaCppConfig(
        gguf_path=cfg["paths"]["gguf_path"],
        n_ctx=int(cfg["llm"].get("n_ctx", 4096)),
        n_threads=int(cfg["llm"].get("n_threads", 4)),
        n_gpu_layers=int(cfg["llm"].get("n_gpu_layers", 0)),
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    llm = LlamaCppWrapper(lcfg)
    prompt = "\n".join([
        f"System: {SYSTEM_MSG}",
        f"User: Question:\n{q}\n\nContext:\n{context}\n",
        "Assistant:"
    ])
    print(prompt)
    answer = llm.chat(prompt).strip()
    # print( )
    console.print("\n" + "="*80)
    console.print(f"[bold]QUESTION:[/bold] {q}")
    console.print("="*80)
    console.print(answer)

    if bool(cfg.get("cli", {}).get("show_sources", True)):
        console.print("\n---\n[bold]Sources:[/bold]")
        for d in docs:
            src = Path(d.metadata.get("source", "")).name
            page = d.metadata.get("page", "n/a")
            console.print(f"- {src} (page {page})")
    console.print("="*80)

if __name__ == "__main__":
    app()
