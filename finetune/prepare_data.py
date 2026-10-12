"""Build a triage-classifier dataset from real reviewer feedback.

Label 1 = the comment was useful (👍 or its thread was resolved).
Label 0 = the comment was not useful (👎).
Comments with no signal, or conflicting signals, are skipped.

Split BY REPOSITORY (not randomly) so the validation score measures generalisation to unseen
codebases rather than memorisation of one project's style.

    python -m finetune.prepare_data --out finetune/data
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from codelens.config import get_settings
from codelens.storage import Storage
from codelens.triage import triage_text

POSITIVE = {"thumbs_up", "resolved"}
NEGATIVE = {"thumbs_down"}


def build_examples(rows: list[dict]) -> list[dict]:
    examples = []
    for r in rows:
        sigs = set(r["signals"])
        pos, neg = bool(sigs & POSITIVE), bool(sigs & NEGATIVE)
        if pos == neg:  # no signal or conflicting signals
            continue
        examples.append({"text": triage_text(r["payload"]), "label": int(pos), "repo": r["repo"],
                         "comment_id": r["gh_comment_id"]})
    return examples


def split_by_repo(examples: list[dict], val_fraction: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_repo[ex["repo"]].append(ex)
    repos = sorted(by_repo)
    rng = random.Random(seed)
    if len(repos) >= 3:
        rng.shuffle(repos)
        n_val = max(1, round(len(repos) * val_fraction))
        val_repos = set(repos[:n_val])
        train = [e for r in repos if r not in val_repos for e in by_repo[r]]
        val = [e for r in val_repos for e in by_repo[r]]
        return train, val
    print("WARNING: fewer than 3 repositories; falling back to a random split. "
          "Validation numbers will be optimistic because of repository leakage.")
    rng.shuffle(examples)
    cut = max(1, int(len(examples) * (1 - val_fraction)))
    return examples[:cut], examples[cut:]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/data")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()
    rows = Storage(get_settings().db_path).labeled_comments()
    examples = build_examples(rows)
    if len(examples) < 100:
        print(f"Only {len(examples)} labeled examples. Fine-tuning needs a few hundred to be meaningful; "
              "keep collecting feedback (or use patch validation instead).")
    train, val = split_by_repo(examples, args.val_fraction)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    pos = sum(e["label"] for e in examples)
    print(f"{len(train)} train / {len(val)} val examples ({pos} positive, {len(examples) - pos} negative)")


if __name__ == "__main__":
    main()
