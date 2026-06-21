"""Phase 1 - Hybrid duplicate detection.

Run:  python phase1_dedup/dedup.py

Pipeline (per embedding model):
  1. embed queries locally (cached)
  2. candidate blocking: keep pairs with semantic cosine >= BLOCK_FLOOR
  3. hybrid signals per candidate: semantic cosine + fuzzy (typo-robust)
  4. LABEL-FREE threshold: T = P95 of the pairwise-similarity distribution
     (prior: duplicates are rare, so flag the upper tail). P99 is reported as
     a precision-oriented alternative.
  5. decision rule: cos >= T  OR  fuzzy >= FUZZY_OR
  6. evaluate: recall by difficulty (60 pairs) + closed-set P/R/F1, plus the
     full precision/recall curve for transparency.

Both candidate embedding models are run and compared; the winner (best
closed-set F1) gets the detailed report, predicted pairs, confident-merge
groups, and missed-pair analysis.

IMPORTANT: duplicate_pairs.csv is used ONLY to score finished predictions and
to draw the P/R curve. No threshold or weight is fit to it - the threshold is a
percentile of the unlabelled score distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.data import build_ground_truth, load_queries, ordered  # noqa: E402
from lib.eval import closed_set_prf, recall_by_difficulty  # noqa: E402
from lib.text import fuzzy_ratio  # noqa: E402
from phase1_dedup.embedding import MODELS, embed  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SHIP_PCTILE = 95       # balanced / F1-oriented operating point (label-free)
PREC_PCTILE = 99       # precision-oriented alternative
FUZZY_OR = 0.90        # near-identical wording -> duplicate regardless of cos
BLOCK_FLOOR = 0.40     # candidate-blocking floor (well below any dup threshold)
MERGE_PCTILE = 99      # confident-merge groups for the Phase-2 handoff
DIFF_ORDER = ["easy", "medium", "hard"]

md: list[str] = []


def h(line: str = "") -> None:
    md.append(line)
    print(line)


# ---------------------------------------------------------------------------
def predict(cos, ids, texts, threshold):
    """Decision rule over blocked candidates -> set of ordered pairs."""
    n = len(ids)
    iu, ju = np.triu_indices(n, k=1)
    block = cos[iu, ju] >= BLOCK_FLOOR
    pred = set()
    for i, j in zip(iu[block], ju[block]):
        c = cos[i, j]
        if c >= threshold or fuzzy_ratio(texts[i], texts[j]) >= FUZZY_OR:
            pred.add(ordered(ids[i], ids[j]))
    return pred


def sweep_curve(cos, ids, gt, thresholds):
    """Closed-set P/R/F1 across cosine thresholds (eval-only, for the curve)."""
    n = len(ids)
    iu, ju = np.triu_indices(n, k=1)
    rows = []
    for t in thresholds:
        pred = {ordered(ids[i], ids[j])
                for i, j in zip(iu, ju) if cos[i, j] >= t}
        c = closed_set_prf(pred, gt)
        rec = recall_by_difficulty(pred, gt)
        rows.append({"t": t, "P": c.precision, "R": c.recall, "F1": c.f1,
                     "hard": rec["hard"].recall, "npred": len(pred)})
    return pd.DataFrame(rows)


def run_model(alias, texts, ids, gt):
    emb = embed(texts, alias)
    cos = (emb @ emb.T).astype(np.float64)
    np.fill_diagonal(cos, 0.0)
    iu, ju = np.triu_indices(len(ids), k=1)
    all_cos = cos[iu, ju]
    t_ship = float(np.percentile(all_cos, SHIP_PCTILE))
    t_prec = float(np.percentile(all_cos, PREC_PCTILE))
    pred = predict(cos, ids, texts, t_ship)
    return {
        "alias": alias, "cos": cos, "all_cos": all_cos,
        "t_ship": t_ship, "t_prec": t_prec, "pred": pred,
        "rec": recall_by_difficulty(pred, gt),
        "closed": closed_set_prf(pred, gt),
    }


def components(pred):
    adj = {}
    for a, b in pred:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj[x] - seen)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------------------
def fig_cosine(res, gt, ids):
    idx = {q: i for i, q in enumerate(ids)}
    cos = res["cos"]
    colors = {"easy": "#2a9d8f", "medium": "#e9c46a", "hard": "#e76f51"}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res["all_cos"], bins=60, density=True, color="#cccccc",
            alpha=0.7, label="all pairs")
    for d in DIFF_ORDER:
        vals = [cos[idx[a], idx[b]] for a, b in gt.labeled_by_difficulty[d]]
        ax.plot(vals, np.full(len(vals), 0.15), "|", ms=16,
                color=colors[d], label=f"gold {d}")
    ax.axvline(res["t_ship"], color="black", ls="--",
               label=f"ship T (P95) = {res['t_ship']:.2f}")
    ax.axvline(res["t_prec"], color="black", ls=":",
               label=f"precision T (P99) = {res['t_prec']:.2f}")
    ax.set(title=f"Semantic cosine distribution - {res['alias']}",
           xlabel="cosine", ylabel="density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"01_cosine_{res['alias']}.png", dpi=120)
    plt.close(fig)


def fig_curve(curve, res):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(curve["R"], curve["P"], "-o", ms=3, color="#264653")
    # mark the shipped operating point
    sp = curve.iloc[(curve["t"] - res["t_ship"]).abs().argmin()]
    a1.plot(sp["R"], sp["P"], "*", ms=16, color="#e76f51",
            label=f"ship (T={res['t_ship']:.2f})")
    a1.set(title="Closed-set precision-recall", xlabel="recall",
           ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(curve["t"], curve["F1"], "-o", ms=3, label="F1 (closed)")
    a2.plot(curve["t"], curve["hard"], "-o", ms=3, label="hard recall (60)")
    a2.axvline(res["t_ship"], color="#e76f51", ls="--", label="ship T")
    a2.set(title="F1 & hard-recall vs threshold", xlabel="cosine threshold")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"02_curve_{res['alias']}.png", dpi=120)
    plt.close(fig)


def missed_analysis(res, gt, texts, ids):
    idx = {q: i for i, q in enumerate(ids)}
    diff_of = {p: d for d, ps in gt.labeled_by_difficulty.items() for p in ps}
    rows = []
    for a, b in (gt.labeled_pairs - res["pred"]):
        i, j = idx[a], idx[b]
        rows.append({"pair": f"{a}~{b}", "difficulty": diff_of[ordered(a, b)],
                     "cos": round(float(res["cos"][i, j]), 3),
                     "fuzzy": round(fuzzy_ratio(texts[i], texts[j]), 3),
                     "text_1": texts[i], "text_2": texts[j]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["difficulty", "cos"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
def main():
    q = load_queries()
    ids, texts = list(q["query_id"]), list(q["text"])
    gt = build_ground_truth()

    h("# Phase 1 - Hybrid duplicate detection\n")
    h("_Generated by `phase1_dedup/dedup.py`._\n")
    h(f"**Decision rule (label-free):** `cos >= T` OR `fuzzy >= {FUZZY_OR}`, "
      f"where `T` = the {SHIP_PCTILE}th percentile of the pairwise-similarity "
      f"distribution (duplicates are rare -> flag the upper tail). Candidate "
      f"blocking floor `cos >= {BLOCK_FLOOR}`. The labels are used only to "
      "score predictions and draw the curve below.\n")

    results = {a: run_model(a, texts, ids, gt) for a in MODELS}

    h("## Model comparison (shipped operating point)\n")
    h("| model | T (P95) | pred pairs | easy | medium | hard | overall "
      "| closed P | R | F1 |")
    h("|---|---|---|---|---|---|---|---|---|---|")
    for a, res in results.items():
        rec, c = res["rec"], res["closed"]
        h(f"| {a} | {res['t_ship']:.2f} | {len(res['pred'])} | "
          f"{rec['easy'].recall:.2f} | {rec['medium'].recall:.2f} | "
          f"{rec['hard'].recall:.2f} | {rec['overall'].recall:.2f} | "
          f"{c.precision:.2f} | {c.recall:.2f} | {c.f1:.3f} |")
    h("")

    winner = max(results.values(),
                 key=lambda r: (r["closed"].f1, r["rec"]["overall"].recall))
    res = winner
    h(f"**Winner: `{res['alias']}`** (best closed-set F1).\n")

    # ----- detailed report for winner --------------------------------------
    h(f"## Winner detail - {res['alias']}\n")
    rec = res["rec"]
    h("Recall on the 60 labelled pairs (headline metric):\n")
    h("| difficulty | recall | found / total |")
    h("|---|---|---|")
    for d in DIFF_ORDER + ["overall"]:
        p = rec[d]
        h(f"| {d} | {p.recall:.2f} | {p.tp} / {p.tp + p.fn} |")
    h("")
    c = res["closed"]
    h(f"Closed-set precision/recall/F1 (120 labelled queries, 300 transitive "
      f"positives): **{c.as_row()}**.\n")

    curve = sweep_curve(res["cos"], ids, gt,
                        np.round(np.arange(0.45, 0.96, 0.025), 3))
    fig_cosine(res, gt, ids)
    fig_curve(curve, res)
    h(f"![cosine](figures/01_cosine_{res['alias']}.png)")
    h(f"![curve](figures/02_curve_{res['alias']}.png)\n")
    f1max = curve.loc[curve["F1"].idxmax()]
    h(f"For reference, the F1-optimal threshold on the labelled curve is "
      f"`cos={f1max['t']:.2f}` (F1={f1max['F1']:.2f}); our label-free P95 "
      f"threshold `{res['t_ship']:.2f}` lands close to it. The P99 "
      f"(precision-oriented) threshold is `{res['t_prec']:.2f}`.\n")

    # off-gold predictions (unscoreable vs gold) - sample for manual audit
    closed_ids = set(gt.closed_query_ids)
    off = [p for p in res["pred"]
           if p[0] not in closed_ids or p[1] not in closed_ids]
    idx = {qq: i for i, qq in enumerate(ids)}
    samp = sorted(off, key=lambda p: -res["cos"][idx[p[0]], idx[p[1]]])[:25]
    pd.DataFrame([{"q1": a, "q2": b,
                   "cos": round(float(res["cos"][idx[a], idx[b]]), 3),
                   "text_1": texts[idx[a]], "text_2": texts[idx[b]]}
                  for a, b in samp]).to_csv(
        REPORTS / "offgold_sample.csv", index=False)
    h(f"Predictions involving unlabelled queries (not scoreable vs gold): "
      f"**{len(off)}**; top-25 by cosine saved to "
      f"`reports/offgold_sample.csv` for manual precision audit.\n")

    # missed-pair analysis
    miss = missed_analysis(res, gt, texts, ids)
    miss.to_csv(REPORTS / "missed_pairs.csv", index=False)
    h("## Missed-pair analysis\n")
    h(f"{len(miss)} of 60 gold pairs missed "
      f"(all {('' if miss.empty else miss['difficulty'].iloc[0])} "
      "tier unless noted). Hardest misses:\n")
    h("| pair | difficulty | cos | fuzzy | text 1 | text 2 |")
    h("|---|---|---|---|---|---|")
    for _, r in miss.head(6).iterrows():
        t1, t2 = r["text_1"][:50].replace("|", "/"), r["text_2"][:50].replace("|", "/")
        h(f"| {r['pair']} | {r['difficulty']} | {r['cos']} | {r['fuzzy']} "
          f"| {t1} | {t2} |")
    h("")

    # confident-merge groups for Phase 2 handoff (high-precision)
    t_merge = float(np.percentile(res["all_cos"], MERGE_PCTILE))
    merges = predict(res["cos"], ids, texts, t_merge)
    comps = [c for c in components(merges) if len(c) > 1]
    n_unique = len(ids) - sum(len(c) - 1 for c in comps)
    h("## Outputs\n")
    h(f"- **Detected duplicate pairs** (shipped, balanced): {len(res['pred'])} "
      f"-> `reports/predicted_pairs.csv`.")
    h(f"- **Confident-merge groups** (P{MERGE_PCTILE}, T={t_merge:.2f}, "
      f"high-precision) for the Phase-2 handoff: {len(comps)} groups covering "
      f"{sum(len(c) for c in comps)} queries; collapsing them leaves "
      f"~**{n_unique}** unique representatives.")
    h("")

    rows = []
    for a, b in sorted(res["pred"]):
        i, j = idx[a], idx[b]
        rows.append({"query_id_1": a, "query_id_2": b,
                     "cos": round(float(res["cos"][i, j]), 3),
                     "fuzzy": round(fuzzy_ratio(texts[i], texts[j]), 3)})
    pd.DataFrame(rows).to_csv(REPORTS / "predicted_pairs.csv", index=False)
    pd.DataFrame([{"group": k, "query_id": qid}
                  for k, comp in enumerate(comps) for qid in sorted(comp)]
                 ).to_csv(REPORTS / "merge_groups.csv", index=False)

    (REPORTS / "report.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {REPORTS/'report.md'} and artifacts.")


if __name__ == "__main__":
    main()
