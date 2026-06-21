"""Thin Anthropic client wrapper: structured JSON calls, on-disk response
caching, token/cost accounting, and a deterministic offline stub.

- Model is env-configurable (`Q19_LLM_MODEL`, default claude-opus-4-8).
- Every call is cached on disk by (model, system, user, schema) so re-runs are
  free and offline. Delete `artifacts/llm_cache/` to force fresh calls.
- `offline=True` (or no ANTHROPIC_API_KEY) returns a deterministic stub built
  from the schema + a caller-supplied hint, so the whole pipeline runs and the
  report renders without network. Stubbed fields are clearly marked.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "artifacts" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("Q19_LLM_MODEL", "claude-opus-4-8")

# USD per 1M tokens (input, output) - from the Anthropic model catalogue.
PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = DEFAULT_MODEL
    per_call: list = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, offline: bool | None = None):
        self.model = model
        if offline is None:
            offline = not os.environ.get("ANTHROPIC_API_KEY")
        self.offline = offline
        self.usage = Usage(model=model)
        self._client = None
        if not offline:
            import anthropic  # lazy
            self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{self.model}_{key}.json"

    def _key(self, system: str, user: str, schema: dict | None) -> str:
        blob = json.dumps([self.model, system, user, schema], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    def json(self, system: str, user: str, schema: dict, *, hint: dict,
             label: str = "", max_tokens: int = 700) -> dict:
        """Return a JSON object matching `schema`. `hint` seeds the offline
        stub so the report still reads sensibly without the API."""
        key = self._key(system, user, schema)
        cache = self._cache_path(key)
        if cache.exists():
            self.usage.calls += 1
            self.usage.cache_hits += 1
            return json.loads(cache.read_text())["data"]

        if self.offline:
            data = dict(hint)
            data["_stub"] = True
            cache.write_text(json.dumps({"data": data, "stub": True}))
            self.usage.calls += 1
            return data

        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(text)
        self._record(resp, label)
        cache.write_text(json.dumps({"data": data, "stub": False}))
        return data

    def text(self, system: str, user: str, *, hint: str, label: str = "",
             max_tokens: int = 2000) -> str:
        key = self._key(system, user, None)
        cache = self._cache_path(key)
        if cache.exists():
            self.usage.calls += 1
            self.usage.cache_hits += 1
            return json.loads(cache.read_text())["data"]

        if self.offline:
            out = hint + "\n\n_(offline stub - run with ANTHROPIC_API_KEY for a real report.)_"
            cache.write_text(json.dumps({"data": out, "stub": True}))
            self.usage.calls += 1
            return out

        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        out = next(b.text for b in resp.content if b.type == "text")
        self._record(resp, label)
        cache.write_text(json.dumps({"data": out, "stub": False}))
        return out

    # ------------------------------------------------------------------
    def _record(self, resp, label: str) -> None:
        u = resp.usage
        self.usage.calls += 1
        self.usage.input_tokens += u.input_tokens
        self.usage.output_tokens += u.output_tokens
        self.usage.per_call.append(
            {"label": label, "in": u.input_tokens, "out": u.output_tokens})
