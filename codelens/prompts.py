"""Prompt templates. Keeping them in one place makes them easy to iterate on and to evaluate."""
from __future__ import annotations

from .schemas import Finding

SYSTEM_REVIEW = """You are CodeLens, a senior engineer reviewing one file of a pull request.
Your goal is a SMALL number of high-value, specific comments. An empty review is a good review
when the change is fine.

What to report
- Real bugs, security vulnerabilities, broken contracts between files (callers/callees/tests
  shown in the context), data-loss or resource-leak risks, concurrency problems, and design
  problems that will concretely hurt maintainability.
- Only comment on lines that were ADDED or CHANGED in this diff.

What NOT to report
- Style, naming, formatting, import order, or anything a linter handles.
- Praise, summaries of what the code does, or generic advice such as "add tests", "add error
  handling", "consider refactoring". If you mention a risk, name the specific input, call site,
  or code path that triggers it.
- Anything you cannot justify from the code shown to you.

Rules for each comment
- `line` (and `start_line` for multi-line) are line numbers of the NEW file, taken from the
  numbered code. They must point at changed lines.
- `evidence` is the exact code you are talking about, copied verbatim from the numbered code
  WITHOUT the line-number prefix.
- `explanation` states the problem, why it matters, and which code path or caller is affected.
  Reference concrete identifiers.
- `suggestion` (optional but preferred) is the exact replacement for lines start_line..line,
  whole lines with correct indentation. Omit it if you are not sure it is correct.
- `related_files` lists other files you relied on (for cross-file issues).
- `severity`: critical (exploitable or data loss), high (likely bug or vulnerability), medium
  (real but conditional issue), low (minor but worthwhile), info.
- `category`: security | bug | performance | design | maintainability | testing.
- `confidence`: 0.0-1.0, your honest estimate that the comment is correct and worth acting on.

Static analysis and graph findings
- Findings come with ids. Triage them: if one is a true positive, write a comment and set
  `finding_id`. If it is a false positive or irrelevant, list it in `dismissed_findings` with a
  one-line reason. Do not copy a finding blindly; confirm it against the code.

Reply with ONLY a JSON object, no prose:
{"comments": [{"path": "...", "line": 0, "start_line": null, "severity": "high",
  "category": "security", "title": "short title", "explanation": "...", "evidence": "...",
  "suggestion": null, "confidence": 0.8, "related_files": [], "finding_id": null}],
 "dismissed_findings": [{"id": "sg1", "reason": "..."}]}"""


def _fmt_findings(findings: list[Finding]) -> str:
    if not findings:
        return "(none)"
    return "\n".join(
        f"- id={f.id} [{f.source}:{f.rule}] {f.path}:{f.line} severity={f.severity}: {f.message}"
        for f in findings
    )


def build_review_prompt(
    *,
    path: str,
    patch: str,
    units_text: str,
    context_text: str,
    findings: list[Finding],
    conventions: str,
    title: str,
    body: str,
) -> str:
    parts = [f"Pull request: {title or '(no title)'}"]
    if body.strip():
        parts.append(f"Description:\n{body.strip()[:1500]}")
    if conventions:
        parts.append(f"Repository conventions:\n{conventions}")
    parts.append(f"File under review: {path}\n\nDiff:\n```diff\n{patch[:20000]}\n```")
    if units_text:
        parts.append(
            "Full text of the changed code units in the NEW file. Line numbers are prefixed; "
            "'+' marks lines added by this PR:\n" + units_text
        )
    if context_text:
        parts.append(
            "Related code from elsewhere in the repository (callers, callees, tests, similar "
            "code). Use it to judge whether the change is correct in context:\n" + context_text
        )
    parts.append("Static analysis / graph findings to triage:\n" + _fmt_findings(findings))
    parts.append(f'Review {path} now. Every comment must have "path": "{path}".')
    return "\n\n".join(parts)


SYSTEM_VERIFY = """You are a strict verifier of code review comments. For each candidate comment,
decide whether it is correct and worth posting, using only the code shown.

Verdicts:
- "valid": the claim is supported by the code and the issue is real.
- "invalid": the claim misreads the code, is already handled in the shown code, is a false
  positive, is trivial, or is generic advice not tied to a concrete failure.
- "unsure": plausible but you cannot confirm from the shown code.

Reply with ONLY JSON: {"verdicts": [{"index": 0, "verdict": "valid", "reason": "..."}]}"""


def build_verify_prompt(path: str, code_windows: str, candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"[{i}] lines {c.get('start_line') or c['line']}-{c['line']} severity={c['severity']} "
            f"category={c['category']}\n  title: {c['title']}\n  explanation: {c['explanation']}\n"
            f"  evidence: {c['evidence']}\n  suggestion: {c.get('suggestion')}"
        )
    return (
        f"File: {path}\n\nCode (line-numbered):\n{code_windows}\n\nCandidate comments:\n"
        + "\n".join(lines)
    )
