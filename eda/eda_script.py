"""Phase 0 - Exploratory data analysis (self-contained).

Run:  python eda/eda_script.py

Produces:
  eda/reports/eda.md          - written findings + tables
  eda/reports/figures/*.png   - supporting charts

This is a standalone review of the data before any modelling. It answers one
design question up front:

  "Which duplicate pairs can lexical methods catch, and which need semantics?"

That answer is what justifies the Phase-1 hybrid (semantic embeddings +
lexical/fuzzy signals).

Ground-truth note: `duplicate_pairs.csv` is EVAL-ONLY. `root_issue` is used
here purely to characterise the data and to define a fair evaluation; it is
never a building signal.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DIFF_ORDER = ["easy", "medium", "hard"]
DIFF_COLOR = {"easy": "#2a9d8f", "medium": "#e9c46a", "hard": "#e76f51"}

# ============================================================================
# Lightweight text helpers (dependency-light; reused as Phase-1 classical
# signals later). Kept inline so Phase 0 is self-contained.
# ============================================================================
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


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    toks = normalize(text).split()
    return [t for t in toks if t not in STOPWORDS] if drop_stop else toks


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


def token_jaccard(t1, t2):
    return jaccard(tokens(t1), tokens(t2))


def char_jaccard(t1, t2, n=3):
    return jaccard(char_ngrams(t1, n), char_ngrams(t2, n))


def fuzzy_ratio(t1, t2):
    return fuzz.token_sort_ratio(normalize(t1), normalize(t2)) / 100.0


# ============================================================================
# Data + ground truth
# ============================================================================
def load_queries() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "queries.csv", dtype=str)
    df["text"] = df["text"].fillna("").str.strip()
    return df


def load_pairs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "duplicate_pairs.csv", dtype=str)


def _ordered(a, b):
    return (a, b) if a <= b else (b, a)


@dataclass
class GroundTruth:
    labeled_pairs: set            # the 60 given pairs
    labeled_by_difficulty: dict   # easy/medium/hard -> pairs
    closed_query_ids: list        # the 120 labelled queries
    closed_positives: set         # transitive same-issue pairs (eval)
    issue_of: dict                # query_id -> root_issue


def build_ground_truth() -> GroundTruth:
    pairs = load_pairs()
    labeled, by_diff, issue_of = set(), {}, {}
    for _, r in pairs.iterrows():
        p = _ordered(r["query_id_1"], r["query_id_2"])
        labeled.add(p)
        by_diff.setdefault(r["difficulty"], set()).add(p)
        issue_of[r["query_id_1"]] = r["root_issue"]
        issue_of[r["query_id_2"]] = r["root_issue"]
    closed_ids = sorted(issue_of)
    closed_pos = {
        _ordered(a, b) for a, b in combinations(closed_ids, 2)
        if issue_of[a] == issue_of[b]
    }
    return GroundTruth(labeled, by_diff, closed_ids, closed_pos, issue_of)


# ============================================================================
# Report builder
# ============================================================================
md: list[str] = []


def h(line: str = "") -> None:
    md.append(line)
    print(line)


def corpus_overview(q: pd.DataFrame) -> None:
    h("## 1. Corpus overview\n")
    n = len(q)
    words = q["text"].str.split().map(len)
    chars = q["text"].str.len()
    exact_dupes = q["text"].str.lower().str.strip().duplicated().sum()
    empties = int((q["text"].str.len() == 0).sum())
    nonascii = int(q["text"].map(lambda s: any(ord(c) > 127 for c in s)).sum())

    h(f"- **{n}** queries, **{q['query_id'].nunique()}** unique ids, "
      f"**{empties}** empty texts.")
    h(f"- Length: **{words.mean():.1f}** words avg "
      f"(min {words.min()}, median {int(words.median())}, max {words.max()}); "
      f"{chars.mean():.0f} chars avg.")
    h(f"- Exact-duplicate texts (case/space-insensitive): **{exact_dupes}**.")
    h(f"- Queries with non-ASCII chars: **{nonascii}**.")
    h("")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(words, bins=30, color="#264653")
    ax[0].set(title="Query length (words)", xlabel="words", ylabel="count")
    ax[1].hist(chars, bins=30, color="#2a9d8f")
    ax[1].set(title="Query length (chars)", xlabel="chars")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_length_dist.png", dpi=120)
    plt.close(fig)
    h("![length](figures/01_length_dist.png)\n")

    short = q.assign(l=chars).nsmallest(3, "l")["text"].tolist()
    long = q.assign(l=chars).nlargest(1, "l")["text"].tolist()
    h("Shortest queries:")
    for t in short:
        h(f"  - `{t}`")
    h("Longest query:")
    h(f"  - `{long[0][:160]}{'…' if len(long[0])>160 else ''}`")
    h("")


def vocabulary(q: pd.DataFrame) -> None:
    h("## 2. Vocabulary & themes\n")
    toks, bigrams = Counter(), Counter()
    for t in q["text"]:
        tt = tokens(t)
        toks.update(tt)
        bigrams.update(" ".join(b) for b in zip(tt, tt[1:]))
    h(f"- Vocabulary (content tokens, stopwords dropped): **{len(toks)}**.")
    h("- Top tokens: " +
      ", ".join(f"`{w}`({c})" for w, c in toks.most_common(12)))
    h("")
    top_bg = bigrams.most_common(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([b for b, _ in top_bg][::-1], [c for _, c in top_bg][::-1],
            color="#457b9d")
    ax.set(title="Top content bigrams", xlabel="count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_bigrams.png", dpi=120)
    plt.close(fig)
    h("![bigrams](figures/02_bigrams.png)\n")


def eval_set_structure() -> None:
    h("## 3. Evaluation set structure (duplicate_pairs.csv)\n")
    pairs = load_pairs()
    gt = build_ground_truth()
    n_cand = len(gt.closed_query_ids) * (len(gt.closed_query_ids) - 1) // 2
    h(f"- **{len(pairs)}** labelled pairs across "
      f"**{pairs['root_issue'].nunique()}** root issues, "
      f"**{pairs['difficulty'].nunique()}** difficulties.")
    h("- Difficulty counts: " +
      ", ".join(f"{d}={int((pairs['difficulty']==d).sum())}" for d in DIFF_ORDER))
    h(f"- Distinct queries referenced: **{len(gt.closed_query_ids)}** "
      f"(6 per root issue, no query in >1 pair).")
    h("")
    h("**Why this matters for evaluation.** Within one root issue all 6 "
      "queries describe the same problem, yet only 3 disjoint pairs are "
      "labelled. A good system *will* flag same-issue pairs that are not in "
      "the 60, so naive precision against the 60 is biased low. We therefore "
      "report two views:")
    h("  - **Headline recall** over the 60 pairs, split easy/medium/hard.")
    h(f"  - **Closed-set precision/recall/F1** on the "
      f"{len(gt.closed_query_ids)} labelled queries, treating same-`root_issue` "
      f"pairs as positive (**{len(gt.closed_positives)}** positives among "
      f"{n_cand} candidate pairs).")
    h("")


def similarity_by_difficulty(q: pd.DataFrame) -> dict:
    h("## 4. Lexical similarity by difficulty (the core finding)\n")
    pairs = load_pairs()
    text_of = dict(zip(q["query_id"], q["text"]))
    vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                          min_df=1, sublinear_tf=True)
    X = vec.fit_transform(q["text"])
    idx = {qid: i for i, qid in enumerate(q["query_id"])}

    rows = []
    for _, r in pairs.iterrows():
        t1, t2 = text_of[r["query_id_1"]], text_of[r["query_id_2"]]
        cos = float(cosine_similarity(X[idx[r["query_id_1"]]],
                                      X[idx[r["query_id_2"]]])[0, 0])
        rows.append({"difficulty": r["difficulty"], "tfidf_cos": cos,
                     "token_jac": token_jaccard(t1, t2),
                     "char_jac": char_jaccard(t1, t2),
                     "fuzzy": fuzzy_ratio(t1, t2)})
    df = pd.DataFrame(rows)
    metrics = ["tfidf_cos", "token_jac", "char_jac", "fuzzy"]

    h("Mean similarity of the **gold positive pairs**, by difficulty:\n")
    tbl = df.groupby("difficulty")[metrics].mean().reindex(DIFF_ORDER).round(3)
    h(tbl.to_markdown())
    h("")

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.6), sharey=True)
    for ax, m in zip(axes, metrics):
        data = [df[df.difficulty == d][m].values for d in DIFF_ORDER]
        bp = ax.boxplot(data, tick_labels=DIFF_ORDER, patch_artist=True,
                        showmeans=True)
        for patch, d in zip(bp["boxes"], DIFF_ORDER):
            patch.set_facecolor(DIFF_COLOR[d])
            patch.set_alpha(0.7)
        ax.set(title=m, ylim=(-0.05, 1.05))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("similarity")
    fig.suptitle("Gold-pair similarity by difficulty (lexical methods)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sim_by_difficulty.png", dpi=120)
    plt.close(fig)
    h("![sim](figures/03_sim_by_difficulty.png)\n")

    h("**Reading it:** easy pairs are lexically obvious; medium degrades "
      "(typos, reordering); hard pairs are near-zero on every lexical metric "
      "— they share *meaning*, not words. This is the empirical case for the "
      "hybrid: lexical/fuzzy for easy+medium, semantic embeddings for hard.\n")
    return {"X": X, "idx": idx}


def separability(q: pd.DataFrame, tfidf: dict) -> None:
    h("## 5. Positive vs negative separability (TF-IDF)\n")
    gt = build_ground_truth()
    X, idx = tfidf["X"], tfidf["idx"]

    pos = {d: [] for d in DIFF_ORDER}
    for d, ps in gt.labeled_by_difficulty.items():
        for a, b in ps:
            pos[d].append(float(cosine_similarity(X[idx[a]], X[idx[b]])[0, 0]))

    rng = np.random.default_rng(19)
    ids = list(q["query_id"])
    gold = gt.labeled_pairs
    neg = []
    while len(neg) < 2000:
        a, b = rng.choice(ids, 2, replace=False)
        if _ordered(a, b) not in gold:
            neg.append(float(cosine_similarity(X[idx[a]], X[idx[b]])[0, 0]))

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(neg, bins=bins, density=True, alpha=0.4, color="#999999",
            label="random non-gold pairs")
    for d in DIFF_ORDER:
        ax.hist(pos[d], bins=bins, density=True, alpha=0.55,
                color=DIFF_COLOR[d], label=f"gold {d}")
    ax.set(title="TF-IDF cosine: gold positives vs random negatives",
           xlabel="cosine similarity", ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_separability.png", dpi=120)
    plt.close(fig)
    h("![sep](figures/04_separability.png)\n")

    h("Single-threshold TF-IDF baseline on the **closed 120-query set** "
      "(preview only; the real Phase-1 model is the hybrid):\n")
    closed = gt.closed_query_ids
    sims, ys = [], []
    for a, b in combinations(closed, 2):
        sims.append(float(cosine_similarity(X[idx[a]], X[idx[b]])[0, 0]))
        ys.append(1 if _ordered(a, b) in gt.closed_positives else 0)
    sims, ys = np.array(sims), np.array(ys)
    h("| threshold | precision | recall | F1 |")
    h("|---|---|---|---|")
    for th in (0.2, 0.3, 0.4, 0.5):
        pred = sims >= th
        tp = int((pred & (ys == 1)).sum())
        fp = int((pred & (ys == 0)).sum())
        fn = int((~pred & (ys == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        h(f"| {th:.1f} | {prec:.2f} | {rec:.2f} | {f1:.2f} |")
    h("\nTakeaway: lexical alone trades precision for recall and cannot reach "
      "the hard tier — confirming we need the semantic leg of the hybrid.\n")


def main() -> None:
    q = load_queries()
    h("# Phase 0 — EDA: banking support queries\n")
    h("_Generated by `eda/eda_script.py`._\n")
    corpus_overview(q)
    vocabulary(q)
    eval_set_structure()
    tfidf = similarity_by_difficulty(q)
    separability(q, tfidf)

    h("## 6. Implications for the pipeline\n")
    h("1. **Phase 1 (hybrid):** semantic embeddings as the recall backbone "
      "(needed for hard), lexical/fuzzy signals to sharpen easy+medium, "
      "kNN blocking to avoid the ~80k all-pairs comparison.")
    h("2. **Evaluation:** headline recall by difficulty on the 60 pairs + "
      "closed-set P/R/F1 via root_issue transitivity; audit a sample of "
      "off-gold predictions for true precision.")
    h("3. **Phase 2 (clustering):** ~20 root issues is a prior, but the full "
      "400 likely span more intents — compare silhouette across several k.")
    h("4. **Phase 3 (LLM):** label clusters (not queries) -> ~k+1 calls; "
      "OpenAI-compatible, configurable, cached client + token/cost estimate.")
    h("")

    out = REPORTS_DIR / "eda.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\nWrote {out} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
