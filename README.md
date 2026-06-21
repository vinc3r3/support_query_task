# Banking support query deduplication, clustering & ops reporting

A three-phase pipeline over 400 real banking support queries:

1. **Duplicate detection** — find queries describing the same underlying issue.
2. **Clustering** — group unique queries by issue type.
3. **Labelling & ops report** — LLM-label each cluster and write a weekly ops report.

> **Status:** all phases complete (0 EDA · 1 dedup · 2 clustering · 3 labelling).

## Layout

Each phase has its own directory and README; shared infra lives in `lib/`.

```
data/           queries.csv, duplicate_pairs.csv (eval only)
lib/            shared infra: data loading, text signals, eval metrics, LLM client
eda/            Phase 0 — data review            (eda/README.md)
phase1_dedup/   Phase 1 — hybrid duplicate detection   (phase1_dedup/README.md)
phase2_cluster/ Phase 2 — clustering by issue type      (phase2_cluster/README.md)
phase3_label/   Phase 3 — LLM labelling & ops report    (phase3_label/README.md)
artifacts/      cached embeddings + LLM responses (gitignored)
```

## Setup & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python eda/eda_script.py             # Phase 0
python phase1_dedup/dedup.py         # Phase 1  (downloads 2 embedding models)
python phase2_cluster/cluster.py     # Phase 2
ANTHROPIC_API_KEY=sk-ant-... python phase3_label/label.py   # Phase 3 (offline stub works without a key)
```

---

## Architecture & key decisions

| Phase | Approach | Why |
|---|---|---|
| **1 — dedup** | Hybrid: bge-small embeddings (semantic) + fuzzy ratio (lexical), kNN-blocked, **label-free** P95 threshold | EDA showed hard pairs share *meaning* not words, so semantics is required; lexical catches typo'd near-duplicates. Threshold can't be fit to the eval set. |
| **2 — cluster** | KMeans on bge-small embeddings, `k` chosen by **silhouette** | Reuses Phase-1 embeddings; silhouette is the standard unsupervised k-selector. Gold labels score the result, never pick `k`. |
| **3 — label** | Anthropic LLM, **one call per cluster** (medoids) + one report call | Labelling clusters not queries keeps cost flat (k+1 calls). Medoids keep prompts small. |

**Cross-cutting:**
- **Embeddings are local** (sentence-transformers) — the provided LLM endpoint is chat-only and unreachable from the dev machine.
- **The eval set is sparse** (60 pairs sample a much larger true-duplicate graph), so precision is measured on a *closed set* of the 120 labelled queries using `root_issue` transitivity — see [`eda/README.md`](eda/README.md).
- **`duplicate_pairs.csv` is eval-only** everywhere: it scores results and draws curves, but never tunes a threshold or selects `k`.

## Evaluation results

**Phase 1 — duplicate detection** (winner: bge-small, balanced operating point)

| metric | result |
|---|---|
| Recall on 60 pairs — easy / medium / hard | **1.00 / 1.00 / 0.20** (overall 0.73) |
| Closed-set precision / recall / F1 | **0.62 / 0.63 / 0.62** |
| Label-free P95 threshold vs F1-optimal | 0.79 vs 0.82 — lands next to optimum |

Easy + medium are effectively solved; the hard tier is intrinsically capped —
those pairs sit *inside* the bulk similarity distribution, indistinguishable
from random different-issue pairs by cosine. All 16 misses are hard pairs
(3+ analysed in the Phase-1 README).

**Phase 2 — clustering**

- **k = 15** (silhouette peak 0.293, a flat plateau ~0.28 across k=12–20).
- **87% of the 60 gold pairs co-cluster** (easy 90% / medium 90% / hard 80%).

**Phase 3 — labelling**

- **16 LLM calls** (15 labels + 1 report), cost from real usage ≈ **$0.01–0.05**
  on claude-opus-4-8 (~5× less on haiku); cached re-runs are free.

## What I'd improve with more time

- **Hard-tier recall** is the main gap. An **LLM-judge / cross-encoder rerank**
  over borderline candidates (cos 0.5–0.79) is the documented path to recover
  hard pairs without wrecking precision.
- **Active threshold/k selection** is deliberately unsupervised; a tiny
  human-labelled dev set (kept separate from the 60) would let us tune properly.
- **HDBSCAN** as a clustering cross-check — auto-k plus a noise class to isolate
  genuinely ambiguous queries instead of forcing them into a cluster.
- **Real ops signals** in Phase 3 (volume spikes, SLA breaches) instead of
  LLM-judged severity from sample text alone.
- **Batch API** for the labelling calls (50% cheaper when latency is fine).
