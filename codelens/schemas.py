"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["info", "low", "medium", "high", "critical"]
Category = Literal[
    "security", "bug", "performance", "design", "maintainability", "testing", "other"
]
SEVERITY_ORDER: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
CATEGORIES = {"security", "bug", "performance", "design", "maintainability", "testing", "other"}


class ReviewComment(BaseModel):
    """One inline review comment. `line` / `start_line` refer to the NEW (head) file."""

    model_config = ConfigDict(extra="ignore")

    path: str
    line: int
    start_line: int | None = None
    severity: Severity = "low"
    category: Category = "other"
    title: str = ""
    explanation: str = ""
    evidence: str = ""
    suggestion: str | None = None
    confidence: float = 0.5
    related_files: list[str] = Field(default_factory=list)
    source: str = "llm"  # llm | static | graph
    finding_id: str | None = None

    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, v: object) -> str:
        v = str(v or "low").lower().strip()
        return v if v in SEVERITY_ORDER else "low"

    @field_validator("category", mode="before")
    @classmethod
    def _category(cls, v: object) -> str:
        v = str(v or "other").lower().strip()
        return v if v in CATEGORIES else "other"

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: object) -> float:
        try:
            return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5

    @field_validator("suggestion", mode="before")
    @classmethod
    def _suggestion(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v)
        return s if s.strip() else None

    @field_validator("related_files", mode="before")
    @classmethod
    def _related(cls, v: object) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

    @field_validator("line", "start_line", mode="before")
    @classmethod
    def _line(cls, v: object) -> int | None:
        if v is None:
            return None
        return int(v)  # type: ignore[arg-type]


class RepoConfig(BaseModel):
    """Per-repository configuration loaded from .codelens.yml."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    min_severity: Severity = "low"
    min_confidence: float = 0.5
    max_comments: int = 8
    max_files: int = 25
    llm_verify: bool = True
    ignore_paths: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    conventions: str = ""

    def category_enabled(self, category: str) -> bool:
        return self.checks.get(category, True)


@dataclass
class ChangedFile:
    path: str
    status: str  # added | modified | removed | renamed
    patch: str | None
    additions: int = 0
    deletions: int = 0


@dataclass
class Finding:
    """A deterministic finding from static analysis or graph queries."""

    id: str
    path: str
    line: int
    end_line: int
    rule: str
    message: str
    severity: str = "medium"
    source: str = "static"  # static | graph
    extra: dict = field(default_factory=dict)


@dataclass
class ReviewInput:
    repo_dir: Path  # checked out at head_sha
    base_sha: str
    head_sha: str
    files: list[ChangedFile]
    title: str = ""
    body: str = ""
    config: RepoConfig = field(default_factory=RepoConfig)
