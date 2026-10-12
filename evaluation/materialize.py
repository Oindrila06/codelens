"""Turn an evaluation case into a checked-out git repo plus the list of changed files."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from codelens.gitutil import diff_files, run_git
from codelens.schemas import RepoConfig, ReviewInput
from codelens.testing import build_repo


def load_cases(directory: Path) -> list[dict]:
    cases = []
    for p in sorted(Path(directory).glob("*.json")):
        case = json.loads(p.read_text())
        case.setdefault("id", p.stem)
        cases.append(case)
    return cases


def materialize(case: dict, workdir: Path) -> ReviewInput:
    """Two case kinds:
      inline: {"files_before": {path: text}, "files_after": {path: text}}
      git:    {"repo_path": "...", "base": "<sha>", "head": "<sha>"}  (local clone is made; the
              original repository is never modified)
    """
    if workdir.exists():
        shutil.rmtree(workdir)
    if case.get("kind", "inline") == "inline":
        repo, base, head = build_repo(workdir, case["files_before"], case["files_after"])
    else:
        workdir.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "-q", "--no-checkout", case["repo_path"], str(workdir)], workdir.parent)
        base, head = case["base"], case["head"]
        run_git(["checkout", "-q", "--detach", head], workdir)
        repo = workdir
    return ReviewInput(
        repo_dir=repo, base_sha=base, head_sha=head, files=diff_files(repo, base, head),
        title=case.get("title", ""), body=case.get("body", ""), config=RepoConfig(),
    )
