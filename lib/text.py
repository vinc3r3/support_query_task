"""Text normalisation and lexical-similarity signals (shared across phases)."""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Tiny, domain-aware stopword list. Aggressive stopwording hurts short queries
# where almost every token carries signal, so we keep it minimal.
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "i", "my", "me", "to", "of", "in",
    "on", "for", "it", "this", "that", "and", "or", "do", "does", "did",
    "how", "can", "could", "would", "will", "with", "at", "be", "been",
    "you", "your", "please", "hi", "hello", "thanks", "thank",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


@lru_cache(maxsize=8192)
def tokens(text: str, drop_stop: bool = True) -> tuple[str, ...]:
    toks = normalize(text).split()
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return tuple(toks)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = normalize(text).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_jaccard(t1: str, t2: str) -> float:
    return jaccard(tokens(t1), tokens(t2))


def char_jaccard(t1: str, t2: str, n: int = 3) -> float:
    return jaccard(char_ngrams(t1, n), char_ngrams(t2, n))


def fuzzy_ratio(t1: str, t2: str) -> float:
    """Token-sort ratio in [0,1]: robust to word order and small typos."""
    return fuzz.token_sort_ratio(normalize(t1), normalize(t2)) / 100.0
