# q19 — Banking support query deduplication, clustering & ops reporting

A three-phase pipeline over 400 real banking support queries:

1. **Duplicate detection** — find queries describing the same underlying issue.
2. **Clustering** — group unique queries by issue type.
3. **Labelling & ops report** — LLM-label each cluster and write a weekly ops report.

> **Status:** Phase 0 (EDA) complete. Phases 1–3 in progress.

## Layout

Each phase is a self-contained directory with its own README.

```
data/    queries.csv, duplicate_pairs.csv (eval only)
eda/     Phase 0 — data review (see eda/README.md)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run a phase, e.g. `python eda/eda_script.py`.

## Results & write-up

Per-phase findings live in each phase's README; consolidated architecture
decisions, evaluation results, and "what I'd improve with more time" are
collected here as the phases land.
