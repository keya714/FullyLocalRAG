# safety.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

DEFAULT_DENY = "I can’t help with that topic. If you have another question, I’m happy to help with safe, allowed topics."

BLOCK_KEYWORDS = {
    "self-harm": ["kill myself", "suicide", "self harm"],
    "explicit_illegal_howto": ["how to make a bomb", "make meth", "credit card skimmer"],
    "malware": ["keylogger", "ransomware", "create a virus"],
    "explosives": ["TATP", "ANFO", "detonator"],
    "hate": ["gas the", "ethnic cleansing"],
}

@dataclass
class GuardConfig:
    enabled: bool = True
    blocked_topics: List[str] = None
    deny_message: str = DEFAULT_DENY

def _contains_any(text: str, phrases: List[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in phrases)

def check_guardrails(query: str, cfg: GuardConfig) -> tuple[bool, str]:
    if not cfg.enabled:
        return False, ""
    topics = cfg.blocked_topics or list(BLOCK_KEYWORDS.keys())
    for t in topics:
        phrases = BLOCK_KEYWORDS.get(t, [])
        if _contains_any(query, phrases):
            return True, (cfg.deny_message or DEFAULT_DENY)
    return False, ""
