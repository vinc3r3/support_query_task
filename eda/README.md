# Phase 0 — EDA

A standalone review of the data before any modelling. It exists to answer one
design question up front:

> Which duplicate pairs can lexical methods catch, and which need semantics?

## Run

```bash
python eda/eda_script.py        # regenerates reports/eda.md + figures/
```

`eda_script.py` has no internal dependencies (only third-party libs), so this
phase stands on its own. Full output: [`reports/eda.md`](reports/eda.md).

## What it looks at

1. **Corpus overview** — 400 queries, length distribution, exact dupes, encoding.
2. **Vocabulary & themes** — top tokens / bigrams.
3. **Eval-set structure** — the 60 labelled pairs (20 root issues × 3 difficulties).
4. **Lexical similarity by difficulty** — TF-IDF cosine, token/char Jaccard, fuzzy.
5. **Separability** — gold positives vs random negatives + a closed-set threshold sweep.

## Key finding

Lexical similarity of gold pairs collapses as difficulty rises — hard pairs
share *meaning*, not words:

| difficulty | TF-IDF cosine | token Jaccard | fuzzy |
|---|---|---|---|
| easy   | 0.61 | 0.61 | 0.79 |
| medium | 0.34 | 0.37 | 0.66 |
| hard   | 0.00 | 0.01 | 0.41 |

![sim](reports/figures/03_sim_by_difficulty.png)

This is the empirical case for a **hybrid** Phase-1 model: semantic embeddings
as the recall backbone (the only thing that reaches the hard tier), with
lexical/fuzzy signals sharpening the easy + medium tiers.

## Evaluation framing (decided here, used by later phases)

`duplicate_pairs.csv` is **eval-only** and is a *sparse sample*: each root issue
contributes 6 queries but only 3 disjoint labelled pairs, so many genuine
same-issue pairs are unlabelled. Scoring precision naively against the 60 pairs
would punish a *correct* system. We therefore use two complementary views:

- **Headline recall** over the 60 labelled pairs, split easy / medium / hard.
- **Closed-set precision / recall / F1** on the 120 labelled queries, treating
  any same-`root_issue` pair as a positive (300 positives / 7,140 candidates).

`root_issue` is used for **evaluation only**, never as a building signal.

## Implications for the pipeline

- **Phase 1 (dedup):** hybrid — embeddings + lexical/fuzzy, kNN blocking to
  avoid the ~80k all-pairs comparison.
- **Phase 2 (clustering):** ~20 root issues is a prior, but the full 400 likely
  span more intents — compare silhouette across several `k`.
- **Phase 3 (LLM):** label clusters (not queries) → ~`k`+1 calls; OpenAI-compatible,
  configurable, cached client + token/cost estimate.
