"""Deterministic verification: every comment must be anchored, grounded and non-generic."""
from __future__ import annotations

import ast
import json
import keyword
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from .diffs import PatchInfo, nearest_commentable, same_hunk
from .schemas import SEVERITY_ORDER, RepoConfig, ReviewComment

_WS = re.compile(r"\s+")
_LINENO = re.compile(r"^\s*\d+\s*[|:+]?\s?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = set(keyword.kwlist) | {"self", "cls", "none", "true", "false", "print", "return", "str", "int"}


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def clean_evidence(evidence: str) -> list[str]:
    """Split evidence into normalized lines, removing line-number or diff-marker prefixes."""
    raw = [ln for ln in evidence.splitlines() if ln.strip()]
    if raw and all(ln.lstrip().startswith("+") for ln in raw):
        raw = [ln.lstrip()[1:] for ln in raw]
    out = []
    for ln in raw:
        stripped = _LINENO.sub("", ln, count=1) if re.match(r"^\s*\d+\s*[|:+]", ln) else ln
        out.append(_norm(stripped))
    return [ln for ln in out if len(ln) >= 3]


def evidence_grounded(evidence: str, file_text: str, patch: str | None) -> bool:
    lines = clean_evidence(evidence)
    if not lines:
        return False
    haystacks = [_norm(file_text)]
    if patch:
        haystacks.append(_norm(patch))
    return any(all(ln in h for ln in lines) for h in haystacks)


def is_generic(c: ReviewComment) -> bool:
    """A comment is generic if it is very short or never mentions any identifier it quotes."""
    if len(c.explanation.strip()) < 40:
        return True
    idents = {m.lower() for m in _IDENT.findall(c.evidence)} - _STOP
    if not idents:
        return False
    text = f"{c.title} {c.explanation}".lower()
    return not any(i in text for i in idents)


def apply_suggestion(file_lines: list[str], start: int, end: int, suggestion: str) -> str:
    new_lines = suggestion.rstrip("\n").split("\n")
    return "\n".join(file_lines[: start - 1] + new_lines + file_lines[end:]) + "\n"


def _ruff_count(text: str, path: str) -> int | None:
    if not shutil.which("ruff"):
        return None
    try:
        proc = subprocess.run(
            ["ruff", "check", "--select", "F,E9", "--isolated", "--no-cache",
             "--output-format", "json", "--stdin-filename", path, "-"],
            input=text, capture_output=True, text=True, timeout=30,
        )
        return len(json.loads(proc.stdout or "[]"))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def suggestion_is_safe(path: str, file_text: str, start: int, end: int, suggestion: str) -> bool:
    """The patched file must still parse and must not introduce new pyflakes errors."""
    lines = file_text.splitlines()
    if start < 1 or end > len(lines) or start > end:
        return False
    original = "\n".join(lines[start - 1 : end])
    if _norm(original) == _norm(suggestion):
        return False
    patched = apply_suggestion(lines, start, end, suggestion)
    if path.endswith(".py"):
        try:
            ast.parse(patched)
        except SyntaxError:
            return False
        before, after = _ruff_count(file_text, path), _ruff_count(patched, path)
        if before is not None and after is not None and after > before:
            return False
    return True


@dataclass
class VerifyStats:
    dropped: list[tuple[ReviewComment, str]]


class Verifier:
    def __init__(
        self,
        file_texts: dict[str, str],
        patches: dict[str, str | None],
        infos: dict[str, PatchInfo],
        cfg: RepoConfig,
        mode: str = "full",
        triage: Callable[[dict], float] | None = None,
        triage_threshold: float = 0.4,
    ) -> None:
        self.file_texts, self.patches, self.infos = file_texts, patches, infos
        self.cfg, self.mode = cfg, mode
        self.triage, self.triage_threshold = triage, triage_threshold
        self.dropped: list[tuple[ReviewComment, str]] = []

    def _drop(self, c: ReviewComment, why: str) -> None:
        self.dropped.append((c, why))

    def check(self, comments: list[ReviewComment]) -> list[ReviewComment]:
        kept: list[ReviewComment] = []
        for c in comments:
            info = self.infos.get(c.path)
            if info is None:
                self._drop(c, "file not in review scope")
                continue
            # 1. Anchor: the comment must sit on a line that exists in the diff.
            snapped = nearest_commentable(c.line, info.commentable)
            if snapped is None:
                self._drop(c, "line not in diff")
                continue
            c.line = snapped
            if c.start_line is not None:
                if (
                    c.start_line >= c.line
                    or c.start_line not in info.commentable
                    or not same_hunk(c.start_line, c.line, info.hunks)
                ):
                    c.start_line, c.suggestion = None, None
            if self.mode == "full":
                if not self._full_checks(c, info):
                    continue
            kept.append(c)
        return self._rank(self._dedupe(kept))

    def _full_checks(self, c: ReviewComment, info: PatchInfo) -> bool:
        text = self.file_texts.get(c.path, "")
        # 2. Grounding: quoted evidence must exist in the file or the diff.
        if not evidence_grounded(c.evidence, text, self.patches.get(c.path)):
            self._drop(c, "evidence not found in code")
            return False
        # 3. Specificity.
        if is_generic(c):
            self._drop(c, "generic comment")
            return False
        # 4. Suggestion must be syntactically valid and not make things worse.
        if c.suggestion is not None:
            start = c.start_line or c.line
            if not suggestion_is_safe(c.path, text, start, c.line, c.suggestion):
                c.suggestion = None
        # 5. Policy from .codelens.yml.
        if c.confidence < self.cfg.min_confidence:
            self._drop(c, f"confidence {c.confidence:.2f} below threshold")
            return False
        if SEVERITY_ORDER[c.severity] < SEVERITY_ORDER[self.cfg.min_severity]:
            self._drop(c, "below min_severity")
            return False
        if not self.cfg.category_enabled(c.category):
            self._drop(c, f"category {c.category} disabled")
            return False
        # 6. Optional fine-tuned triage model.
        if self.triage is not None:
            try:
                p = self.triage(c.model_dump())
            except Exception:  # noqa: BLE001
                p = 1.0
            if p < self.triage_threshold:
                self._drop(c, f"triage model p={p:.2f}")
                return False
        return True

    def _dedupe(self, comments: list[ReviewComment]) -> list[ReviewComment]:
        best: dict[tuple[str, int], ReviewComment] = {}
        for c in comments:
            key = (c.path, c.line)
            cur = best.get(key)
            if cur is None or (SEVERITY_ORDER[c.severity], c.confidence) > (
                SEVERITY_ORDER[cur.severity], cur.confidence,
            ):
                if cur is not None:
                    self._drop(cur, "duplicate line")
                best[key] = c
            else:
                self._drop(c, "duplicate line")
        return list(best.values())

    def _rank(self, comments: list[ReviewComment]) -> list[ReviewComment]:
        ranked = sorted(comments, key=lambda c: (-SEVERITY_ORDER[c.severity], -c.confidence, c.path, c.line))
        for c in ranked[self.cfg.max_comments :]:
            self._drop(c, "over max_comments")
        return ranked[: self.cfg.max_comments]
