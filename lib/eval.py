"""Evaluation helpers for pairwise duplicate detection (shared)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .data import GroundTruth, ordered


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_row(self) -> str:
        return (f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
                f"(tp={self.tp} fp={self.fp} fn={self.fn})")


def prf(pred: set, gold: set) -> PRF:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(p, r, f1, tp, fp, fn)


def recall_by_difficulty(pred: set, gt: GroundTruth) -> dict[str, PRF]:
    """Recall over the 60 labelled pairs, per difficulty (+ overall)."""
    out = {}
    for d, gold in gt.labeled_by_difficulty.items():
        out[d] = prf(pred & gold, gold)  # restrict to this tier's pairs
    out["overall"] = prf(pred & gt.labeled_pairs, gt.labeled_pairs)
    return out


def closed_set_prf(pred: set, gt: GroundTruth) -> PRF:
    """P/R/F1 on the closed 120-query universe using transitive positives.

    Only pairs whose *both* endpoints are labelled queries are scored, so
    predictions involving unlabelled queries neither help nor hurt here.
    """
    closed = set(gt.closed_query_ids)
    universe = {ordered(a, b) for a, b in combinations(gt.closed_query_ids, 2)}
    pred_closed = {p for p in pred if p[0] in closed and p[1] in closed} & universe
    return prf(pred_closed, gt.closed_positives)
