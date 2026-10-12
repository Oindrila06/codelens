"""Collect lightweight repository conventions to show the reviewer."""
from __future__ import annotations

import re
from pathlib import Path

from .schemas import RepoConfig

MAX_CHARS = 2500


def load_conventions(repo_dir: Path, cfg: RepoConfig) -> str:
    parts: list[str] = []
    if cfg.conventions.strip():
        parts.append("Team conventions (from .codelens.yml):\n" + cfg.conventions.strip())
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(errors="replace")
        m = re.search(r"^\[tool\.(ruff|black|flake8|isort|mypy)[^\n]*\n(?:(?!^\[).*\n?)*", text, re.M)
        if m:
            parts.append("Tooling config (pyproject.toml):\n" + m.group(0).strip()[:600])
    for name in ("CONTRIBUTING.md", "CONTRIBUTING.rst"):
        p = repo_dir / name
        if p.is_file():
            parts.append(f"{name} (excerpt):\n" + p.read_text(errors="replace")[:900].strip())
            break
    return "\n\n".join(parts)[:MAX_CHARS]
