"""Thin wrappers around the git CLI."""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

from .diffs import HUNK_RE  # noqa: F401  (re-exported for convenience)
from .schemas import ChangedFile

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "site-packages", "build", "dist",
    "vendor", "third_party", "__pycache__", ".tox", ".mypy_cache", ".ruff_cache",
}


class GitError(RuntimeError):
    pass


def run_git(args: list[str], cwd: Path, check: bool = True, env_extra: dict | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env_extra or {})}
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, env=env, errors="replace"
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def _auth_args(token: str | None) -> list[str]:
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.extraheader=AUTHORIZATION: basic {basic}"]


def ensure_checkout(
    clone_url: str,
    dest: Path,
    head_sha: str,
    base_sha: str | None = None,
    pr_number: int | None = None,
    token: str | None = None,
) -> Path:
    """Fetch just the commits we need (shallow) and check out head_sha, detached.

    The token is passed via an HTTP header, so it never appears in the remote URL or in logs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        run_git(["init", "-q"], dest)
        run_git(["remote", "add", "origin", clone_url], dest)
    else:
        run_git(["remote", "set-url", "origin", clone_url], dest)
    auth = _auth_args(token)
    for sha in [s for s in (head_sha, base_sha) if s]:
        try:
            run_git([*auth, "fetch", "--depth", "1", "-q", "origin", sha], dest)
        except GitError:
            if sha == head_sha and pr_number is not None:
                run_git([*auth, "fetch", "--depth", "1", "-q", "origin", f"pull/{pr_number}/head"], dest)
            else:
                raise
    run_git(["checkout", "-q", "-f", "--detach", head_sha], dest)
    run_git(["clean", "-fdq"], dest)
    return dest


def read_file_at(repo_dir: Path, sha: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{sha}:{path}"], cwd=repo_dir, capture_output=True, errors="replace",
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def list_python_files(repo_dir: Path, max_bytes: int = 300_000) -> list[str]:
    """Tracked .py files, falling back to a directory walk outside git."""
    paths: list[str] = []
    try:
        out = run_git(["ls-files", "-z", "--", "*.py"], repo_dir)
        paths = [p for p in out.split("\0") if p]
    except (GitError, FileNotFoundError):
        pass
    if not paths:
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.endswith(".py"):
                    paths.append((Path(root) / f).relative_to(repo_dir).as_posix())
    good = []
    for p in paths:
        parts = set(Path(p).parts)
        if parts & SKIP_DIRS:
            continue
        fp = repo_dir / p
        try:
            if fp.is_file() and fp.stat().st_size <= max_bytes:
                good.append(p)
        except OSError:
            continue
    return sorted(good)


def diff_files(repo_dir: Path, base: str, head: str, context: int = 3) -> list[ChangedFile]:
    """Build the same per-file structure GitHub's 'list PR files' API returns, locally."""
    names = run_git(["diff", "--name-status", "--no-renames", base, head], repo_dir)
    files: list[ChangedFile] = []
    for line in names.splitlines():
        status_code, _, path = line.partition("\t")
        if not path:
            continue
        status = {"A": "added", "D": "removed", "M": "modified"}.get(status_code[:1], "modified")
        raw = run_git(["diff", f"-U{context}", "--no-renames", base, head, "--", path], repo_dir)
        patch = None
        lines = raw.splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("@@"):
                patch = "\n".join(lines[i:])
                break
        adds = sum(1 for ln in (patch or "").splitlines() if ln.startswith("+"))
        dels = sum(1 for ln in (patch or "").splitlines() if ln.startswith("-"))
        files.append(ChangedFile(path=path, status=status, patch=patch, additions=adds, deletions=dels))
    return files
