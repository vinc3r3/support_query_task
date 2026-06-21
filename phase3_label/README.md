# Phase 3 — LLM labelling & weekly ops report

Use an LLM to label each cluster and generate a short weekly ops report.

## Run

```bash
# Offline stub — runs with no key, deterministic keyword labels:
python phase3_label/label.py

# Real labels (needs an Anthropic API key):
ANTHROPIC_API_KEY=sk-ant-... python phase3_label/label.py

# Cheaper model (~5x less):
Q19_LLM_MODEL=claude-haiku-4-5 ANTHROPIC_API_KEY=sk-ant-... python phase3_label/label.py
```

Full output: [`reports/report.md`](reports/report.md) (+ `cluster_labels.csv`,
`ops_report.md`).

## Approach

- **LLM client** (`lib/llm.py`): official Anthropic SDK, model from
  `Q19_LLM_MODEL` (default `claude-opus-4-8`), structured JSON via
  `output_config.format`, **on-disk response cache** (re-runs are free), and a
  **deterministic offline stub** so the pipeline runs without a key.
- **Why this endpoint:** the task's Gemma endpoint is on a private network
  unreachable from the dev machine, so we target the Anthropic API instead. The
  client is model-agnostic; point `Q19_LLM_MODEL` / `ANTHROPIC_BASE_URL` at any
  OpenAI-/Anthropic-compatible backend.
- **Labelling:** one call per cluster. We send the cluster's **medoids** (the
  `N_REPR=8` queries closest to the centroid), not the whole cluster, so prompts
  stay small. The model returns `label`, `description`, `severity`, and
  `needs_escalation` (fraud / lost access / missing money → escalate).
- **Ops report:** one call. Cluster labels + volumes are aggregated in code,
  then the LLM writes the narrative (top issues, escalations, 2 recommendations).

## LLM call count & cost

- **Total calls = k + 1 = 16** (15 cluster labels + 1 ops report) — fixed,
  independent of corpus size, because we label *clusters* not queries.
- Cost is computed from **real token usage** × the model's catalogue price.
  At ~16 small calls the cost is ~**$0.01–0.05** on `claude-opus-4-8` and
  ~5× less on `claude-haiku-4-5`. Caching makes every re-run **$0.00**.

## Tradeoffs & what I'd improve

- **Medoids over full clusters** trades a little context for a flat, predictable
  token cost; a noisy cluster could be mislabelled from an unrepresentative
  centroid. A cheap guard is to also send the 2–3 lowest-similarity members.
- **Severity/escalation is LLM-judged** from sample text only — it has no real
  signal of volume spikes or SLA breaches; wire in ticket metadata for a real
  ops system.
- **Batch API** would halve cost for the labelling calls if latency is not a
  concern.
- The offline stub keeps the repo runnable and CI-testable, but its labels are
  keyword bags — always run with a key for the deliverable.
