"""Local sentence-embeddings with on-disk caching.

The provided LLM endpoint is chat-only, so embeddings are computed locally.
Embeddings are cached per (model, corpus) so re-runs are instant and offline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "artifacts"
CACHE_DIR.mkdir(exist_ok=True)

# short alias -> HuggingFace id
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def _corpus_key(texts: list[str]) -> str:
    h = hashlib.sha1("␟".join(texts).encode("utf-8")).hexdigest()[:12]
    return h


def embed(texts: list[str], model_alias: str) -> np.ndarray:
    """Return L2-normalised embeddings (n, d) for `texts`, cached on disk."""
    model_id = MODELS[model_alias]
    cache = CACHE_DIR / f"emb_{model_alias}_{_corpus_key(texts)}.npy"
    if cache.exists():
        return np.load(cache)

    from sentence_transformers import SentenceTransformer  # lazy import

    # bge models recommend a query prefix for retrieval; for symmetric
    # similarity over homogeneous short queries we embed raw text for both.
    model = SentenceTransformer(model_id)
    emb = model.encode(texts, normalize_embeddings=True,
                       show_progress_bar=False, batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache, emb)
    return emb
