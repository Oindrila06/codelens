"""Build an evaluation case from a security/bug FIX commit in a local git repository.

Idea: review the *reverse* of the fix. Base = the fixed commit, head = its parent (the vulnerable
code). A good reviewer should flag the lines that the fix removed. This turns any real CVE-fix or
bug-fix commit into a labeled pull request without hand-labeling.

    python -m evaluation.make_case_from_fix --repo ~/src/flask --fix <sha> --id flask-1234 \
        --category security --title "Revert sanitisation" --out evaluation/cases/flask-1234.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codelens.diffs import analyze_patch, contiguous_ranges
from codelens.gitutil import diff_files, run_git


def build_case(repo: Path, fix: str, case_id: str, category: str, title: str) -> dict:
    fix_sha = run_git(["rev-parse", fix], repo).strip()
    parent = run_git(["rev-parse", f"{fix_sha}^"], repo).strip()
    files = [f for f in diff_files(repo, fix_sha, parent, context=3) if f.path.endswith(".py") and f.patch]
    expected = []
    for f in files:
        for start, end in contiguous_ranges(analyze_patch(f.patch).added):
            expected.append({"path": f.path, "start_line": start, "end_line": end, "category": category})
    return {
        "id": case_id, "kind": "git", "repo_path": str(repo), "base": fix_sha, "head": parent,
        "title": title, "expected": expected,
        "needed_context": [],  # fill in: files a reviewer must read to see why this is a bug
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--fix", required=True, help="SHA of the fix commit")
    ap.add_argument("--id", required=True)
    ap.add_argument("--category", default="security")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    case = build_case(Path(args.repo).expanduser(), args.fix, args.id, args.category, args.title or args.id)
    Path(args.out).write_text(json.dumps(case, indent=2))
    print(f"Wrote {args.out} with {len(case['expected'])} expected issue range(s). "
          "Review it and fill in needed_context by hand.")


if __name__ == "__main__":
    main()
