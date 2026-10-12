"""LLM review and LLM verification calls."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from .llm import LLM
from .prompts import SYSTEM_REVIEW, SYSTEM_VERIFY, build_review_prompt, build_verify_prompt
from .schemas import Finding, ReviewComment

log = logging.getLogger(__name__)


def number_lines(text: str, start: int, added: set[int] | None = None) -> str:
    out = []
    for i, line in enumerate(text.splitlines()):
        n = start + i
        mark = "+" if added and n in added else " "
        out.append(f"{n:>5} {mark} {line}")
    return "\n".join(out)


class Reviewer:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def review_file(
        self,
        *,
        path: str,
        patch: str,
        units_text: str,
        context_text: str,
        findings: list[Finding],
        conventions: str,
        title: str,
        body: str,
    ) -> tuple[list[ReviewComment], list[dict]]:
        user = build_review_prompt(
            path=path, patch=patch, units_text=units_text, context_text=context_text,
            findings=findings, conventions=conventions, title=title, body=body,
        )
        try:
            raw = self.llm.complete_json(SYSTEM_REVIEW, user)
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the whole review
            log.warning("LLM review failed for %s: %s", path, exc)
            return [], []
        by_id = {f.id: f for f in findings}
        comments: list[ReviewComment] = []
        for item in raw.get("comments", []) or []:
            if not isinstance(item, dict):
                continue
            item.setdefault("path", path)
            try:
                c = ReviewComment(**item)
            except (ValidationError, TypeError, ValueError):
                continue
            if c.path != path:
                c.path = path
            finding = by_id.get(c.finding_id or "")
            c.source = finding.source if finding else "llm"
            comments.append(c)
        dismissed = [d for d in (raw.get("dismissed_findings") or []) if isinstance(d, dict)]
        return comments, dismissed

    def verify_comments(
        self, path: str, code_windows: str, comments: list[ReviewComment]
    ) -> tuple[list[ReviewComment], list[tuple[ReviewComment, str]]]:
        """Second-opinion pass: drop comments the verifier marks invalid."""
        if not comments:
            return [], []
        payload = [c.model_dump() for c in comments]
        try:
            raw = self.llm.complete_json(SYSTEM_VERIFY, build_verify_prompt(path, code_windows, payload))
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM verification failed for %s: %s", path, exc)
            return comments, []
        verdicts = {}
        for v in raw.get("verdicts", []) or []:
            if isinstance(v, dict) and isinstance(v.get("index"), int):
                verdicts[v["index"]] = v
        kept, dropped = [], []
        for i, c in enumerate(comments):
            v = verdicts.get(i)
            if v and str(v.get("verdict", "")).lower() == "invalid":
                dropped.append((c, f"llm-verify: {v.get('reason', 'invalid')}"))
            else:
                if v and str(v.get("verdict", "")).lower() == "unsure":
                    c.confidence = min(c.confidence, 0.5)
                kept.append(c)
        return kept, dropped
