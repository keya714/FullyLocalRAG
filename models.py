# models.py
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass

import os
import ollama  # pip install ollama

# ---- Ollama backend ----
@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"  # any local ollama tag you have pulled
    num_ctx: int = 4096
    temperature: float = 0.2
    max_new_tokens: int = 320
    stop: tuple[str, ...] = ()

class OllamaWrapper:
    def __init__(self, cfg: OllamaConfig):
        # Assumes `ollama serve` is running locally (default: http://localhost:11434)
        self.cfg = cfg

    def chat(self, prompt: str) -> str:
        # We keep your existing prompt-builder; send it as a single prompt to Ollama.
        options = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
            "num_predict": self.cfg.max_new_tokens,
        }
        # Ollama expects a list for stop
        if self.cfg.stop:
            options["stop"] = list(self.cfg.stop)

        out = ollama.generate(
            model=self.cfg.model,
            prompt=prompt,
            options=options,
        )
        return out["response"]

# (Optional helper kept as-is if you still need it elsewhere)
def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    out = []
    for m in messages:
        role = m["role"]
        out.append(f"<|{role}|>\n{m['content']}\n")
    out.append("<|assistant|>\n")
    return "\n".join(out)

# ---- llama.cpp backend ----
@dataclass
class LlamaCppConfig:
    gguf_path: str
    n_ctx: int = 4096
    n_threads: int = max(1, os.cpu_count() // 2)
    n_gpu_layers: int = 0
    temperature: float = 0.2
    max_new_tokens: int = 320
    stop: tuple[str, ...] = ()

class LlamaCppWrapper:
    def __init__(self, cfg: LlamaCppConfig):
        from llama_cpp import Llama
        if not os.path.exists(cfg.gguf_path):
            raise FileNotFoundError(f"GGUF not found: {cfg.gguf_path}")
        self.llm = Llama(
            model_path=cfg.gguf_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        self.cfg = cfg

    def chat(self, prompt: str) -> str:
        # Hard cap prompt to leave space for generation
        # (very rough: reserve 1/3 of context for output)
        try:
            self.llm.reset()
        except Exception:
            pass

        max_prompt_tokens = int(self.cfg.n_ctx * 2 / 3)
        if len(prompt) > max_prompt_tokens:
            prompt = prompt[-max_prompt_tokens:]

        out = self.llm(
            prompt,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            stop=self.cfg.stop if self.cfg.stop else None,
        )
        return out["choices"][0]["text"]

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    # Fallback simple template
    out = []
    for m in messages:
        role = m["role"]
        out.append(f"<|{role}|>\n{m['content']}\n")
    out.append("<|assistant|>\n")
    return "\n".join(out)