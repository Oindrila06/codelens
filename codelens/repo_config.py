"""Load per-repo .codelens.yml and decide which paths are in scope."""
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import RepoConfig

log = logging.getLogger(__name__)
CONFIG_NAMES = (".codelens.yml", ".codelens.yaml")


def load_repo_config(repo_dir: Path) -> RepoConfig:
    for name in CONFIG_NAMES:
        p = repo_dir / name
        if p.is_file():
            try:
                data = yaml.safe_load(p.read_text()) or {}
                return RepoConfig(**data)
            except (yaml.YAMLError, ValidationError, TypeError) as exc:
                log.warning("Invalid %s (%s); using defaults", name, exc)
    return RepoConfig()


def is_ignored(path: str, patterns: list[str]) -> bool:
    # fnmatch's "*" also matches "/", so "docs/*" ignores everything below docs/.
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)
