# models.py
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass

from transformers import AutoTokenizer, pipeline
import os

@dataclass
class HFConfig:
    model_id: str
    device_map: str = "auto"
    torch_dtype: str | None = "auto"
    temperature: float = 0.2
    max_new_tokens: int = 320
    do_sample: bool = False
    quantization: str = "int4"  # none | int8 | int4

def load_transformers_chat(cfg: HFConfig):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = None
    if cfg.quantization in {"int8", "int4"}:
        load_in_8bit = cfg.quantization == "int8"
        load_in_4bit = cfg.quantization == "int4"
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True if load_in_4bit else None,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None,
            bnb_4bit_compute_dtype=None,
        )

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map=cfg.device_map,
        torch_dtype=cfg.torch_dtype if cfg.torch_dtype != "auto" else None,
        quantization_config=bnb_config,
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map=cfg.device_map,
        torch_dtype=cfg.torch_dtype if cfg.torch_dtype != "auto" else None,
        return_full_text=False,
    )
    return tok, gen


# ---- llama.cpp backend ----
@dataclass
class LlamaCppConfig:
    gguf_path: str
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.2
    max_new_tokens: int = 320

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
        out = self.llm(
            prompt,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            stop=None,
        )
        print(out)
        print("&"*40)
        return out["choices"][0]["text"]

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    # Prefer the model's template if available
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
    print(out)
    print("*"*40)
    return "\n".join(out)
