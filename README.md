# q19 — Banking support query deduplication, clustering & ops reporting

A three-phase pipeline over 400 real banking support queries:

1. **Duplicate detection** — find queries describing the same underlying issue.
2. **Clustering** — group unique queries by issue type.
3. **Labelling & ops report** — LLM-label each cluster and write a weekly ops report.

> Status: **Phase 0 (EDA) complete.** Phases 1–3 are being built step by step.

## Layout

Each phase is a self-contained directory. Phase 0 (EDA) is done; later phases
will be added the same way (`phase1_dedup/`, `phase2_cluster/`, …).

```
data/                 queries.csv, duplicate_pairs.csv (eval only)
eda/                  Phase 0 — data review
  eda_script.py       standalone EDA (no internal deps)
  reports/            generated eda.md + figures/
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eda/eda_script.py        # regenerates eda/reports/eda.md + figures
```

## Evaluation framing (decided up front)

`duplicate_pairs.csv` is **eval-only** and is a *sparse sample*: each of the 20
root issues contributes 6 queries but only 3 disjoint labelled pairs, so many
genuine same-issue pairs are unlabelled. Scoring precision naively against the
60 pairs therefore punishes a *correct* system. We report two complementary
views:

- **Headline recall** over the 60 labelled pairs, split easy / medium / hard.
- **Closed-set precision / recall / F1** on the 120 labelled queries, treating
  any same-`root_issue` pair as a positive (300 positives / 7,140 candidates).
  `root_issue` is used for evaluation only, never as a building signal.

## Key EDA finding

Lexical similarity of gold pairs collapses with difficulty — hard pairs share
*meaning*, not words (TF-IDF cosine: easy **0.61**, medium **0.34**, hard
**0.00**). This is the empirical case for a **hybrid** Phase-1 model: semantic
embeddings for recall (esp. hard), lexical/fuzzy signals to sharpen easy+medium.
See [`eda/reports/eda.md`](eda/reports/eda.md).

## Architecture decisions, tradeoffs, full eval results, and "what I'd improve"

_To be completed as Phases 1–3 land._
