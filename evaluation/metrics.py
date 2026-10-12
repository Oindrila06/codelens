"""Scoring: match predicted comments to labeled expected issues and aggregate."""
from __future__ import annotations

from dataclasses import dataclass

from codelens.schemas import ReviewComment

LINE_TOLERANCE = 2


@dataclass
class CaseScore:
    tp: int
    fp: int
    fn: int
    n_comments: int
    retrieval_recall: float | None = None


def _span(c: ReviewComment) -> tuple[int, int]:
    return (c.start_line or c.line, c.line)


def matches(pred: ReviewComment, exp: dict, tol: int = LINE_TOLERANCE) -> bool:
    if pred.path != exp["path"]:
        return False
    lo, hi = _span(pred)
    return lo - tol <= exp["end_line"] and exp["start_line"] <= hi + tol


def score_case(comments: list[ReviewComment], expected: list[dict]) -> CaseScore:
    """One-to-one greedy matching. Unmatched predictions are false positives."""
    used: set[int] = set()
    tp = 0
    for exp in expected:
        for i, c in enumerate(comments):
            if i not in used and matches(c, exp):
                used.add(i)
                tp += 1
                break
    return CaseScore(tp=tp, fp=len(comments) - len(used), fn=len(expected) - tp, n_comments=len(comments))


def retrieval_recall(context_paths: set[str], needed: list[str]) -> float | None:
    if not needed:
        return None
    return sum(1 for p in needed if p in context_paths) / len(needed)


def aggregate(scores: list[CaseScore]) -> dict:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fp + fn) else None)
    rr = [s.retrieval_recall for s in scores if s.retrieval_recall is not None]
    return {
        "cases": len(scores), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "comments_per_pr": sum(s.n_comments for s in scores) / len(scores) if scores else 0.0,
        "retrieval_recall": sum(rr) / len(rr) if rr else None,
    }
