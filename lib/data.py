"""Data loading and ground-truth construction (shared across phases).

`duplicate_pairs.csv` is EVAL-ONLY. `root_issue` is used to define a fair
evaluation (closed-set transitive positives) but is NEVER a building signal.
See eda/README.md for the rationale behind the two ground-truth views.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_queries() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "queries.csv", dtype=str)
    df["text"] = df["text"].fillna("").str.strip()
    return df


def load_pairs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "duplicate_pairs.csv", dtype=str)


def ordered(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass
class GroundTruth:
    labeled_pairs: set                 # the 60 given pairs
    labeled_by_difficulty: dict        # easy/medium/hard -> set of pairs
    closed_query_ids: list             # the 120 labelled queries
    closed_positives: set              # transitive same-issue pairs (eval)
    issue_of: dict                     # query_id -> root_issue


def build_ground_truth() -> GroundTruth:
    pairs = load_pairs()
    labeled, by_diff, issue_of = set(), {}, {}
    for _, r in pairs.iterrows():
        p = ordered(r["query_id_1"], r["query_id_2"])
        labeled.add(p)
        by_diff.setdefault(r["difficulty"], set()).add(p)
        issue_of[r["query_id_1"]] = r["root_issue"]
        issue_of[r["query_id_2"]] = r["root_issue"]
    closed_ids = sorted(issue_of)
    closed_pos = {
        ordered(a, b) for a, b in combinations(closed_ids, 2)
        if issue_of[a] == issue_of[b]
    }
    return GroundTruth(labeled, by_diff, closed_ids, closed_pos, issue_of)
