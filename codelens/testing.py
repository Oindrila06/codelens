"""Helpers for building throwaway git repositories (used by tests and the eval harness)."""
from __future__ import annotations

from pathlib import Path

from .gitutil import run_git

_ID = ["-c", "user.email=codelens@example.com", "-c", "user.name=CodeLens", "-c", "commit.gpgsign=false"]


def _write_tree(repo: Path, files: dict[str, str]) -> None:
    run_git(["rm", "-rq", "--ignore-unmatch", "."], repo, check=False)
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    run_git(["add", "-A"], repo)


def build_repo(root: Path, before: dict[str, str], after: dict[str, str]) -> tuple[Path, str, str]:
    """Create a repo with a base commit (`before`) and a head commit (`after`)."""
    root.mkdir(parents=True, exist_ok=True)
    run_git(["init", "-q", "-b", "main"], root)
    _write_tree(root, before)
    run_git([*_ID, "commit", "-q", "--allow-empty", "-m", "base"], root)
    base = run_git(["rev-parse", "HEAD"], root).strip()
    _write_tree(root, after)
    run_git([*_ID, "commit", "-q", "--allow-empty", "-m", "head"], root)
    head = run_git(["rev-parse", "HEAD"], root).strip()
    return root, base, head
