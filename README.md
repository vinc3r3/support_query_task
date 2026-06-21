# q19 — Banking support query deduplication, clustering & ops reporting

A three-phase pipeline over 400 real banking support queries:

1. **Duplicate detection** — find queries describing the same underlying issue.
2. **Clustering** — group unique queries by issue type.
3. **Labelling & ops report** — LLM-label each cluster and write a weekly ops report.

> **Status:** Phases 0 (EDA), 1 (dedup) and 2 (clustering) complete. Phase 3 in progress.

## Layout

Each phase has its own directory and README; shared infra lives in `lib/`.

```
data/           queries.csv, duplicate_pairs.csv (eval only)
lib/            shared infra: data loading, text signals, eval metrics
eda/            Phase 0 — data review (see eda/README.md)
phase1_dedup/   Phase 1 — hybrid duplicate detection (see its README.md)
phase2_cluster/ Phase 2 — clustering by issue type (see its README.md)
artifacts/      cached embeddings (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run a phase, e.g. `python eda/eda_script.py` or `python phase1_dedup/dedup.py`.

## Results & write-up

Per-phase findings live in each phase's README; consolidated architecture
decisions, evaluation results, and "what I'd improve with more time" are
collected here as the phases land.
