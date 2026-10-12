"""The review pipeline. GitHub-agnostic: it takes a checked-out repo plus a list of changed files,
so the same code path serves the live GitHub App and the offline evaluation harness."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .conventions import load_conventions
from .diffs import PatchInfo, analyze_patch
from .graph import detect_signature_changes
from .indexer.context import build_context
from .indexer.embeddings import get_embedder
from .indexer.index import RepoIndex, StoredChunk
from .indexer.retrieval import Retriever
from .llm import LLM, Usage
from .repo_config import is_ignored
from .reviewer import Reviewer, number_lines
from .schemas import Finding, ReviewComment, ReviewInput
from .static_analysis import analyze, only_on_lines
from .verifier import Verifier

log = logging.getLogger(__name__)
MAX_UNITS_CHARS = 24_000
MAX_UNIT_LINES = 150


@dataclass
class PipelineOptions:
    use_context: bool = True
    use_static: bool = True
    use_graph: bool = True
    llm_review: bool = True
    verify: str = "full"  # "basic" (anchoring only) | "full"
    llm_verify: bool | None = None  # None -> follow .codelens.yml


@dataclass
class ReviewResult:
    comments: list[ReviewComment] = field(default_factory=list)
    dropped: list[tuple[ReviewComment, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    context_paths: set[str] = field(default_factory=set)
    static_tool: str = ""
    index_stats: dict = field(default_factory=dict)
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)
    files_reviewed: int = 0


def _select_files(inp: ReviewInput, settings: Settings, skipped: list[str]):
    cfg = inp.config
    limit = min(cfg.max_files, settings.max_pr_files)
    candidates = []
    for f in inp.files:
        if f.status == "removed":
            continue
        if not f.path.endswith(".py"):
            skipped.append(f"{f.path}: not a Python file")
        elif is_ignored(f.path, cfg.ignore_paths):
            skipped.append(f"{f.path}: ignored by .codelens.yml")
        elif not f.patch:
            skipped.append(f"{f.path}: no textual diff (binary or too large)")
        else:
            candidates.append(f)
    candidates.sort(key=lambda f: -f.additions)
    for f in candidates[limit:]:
        skipped.append(f"{f.path}: over the {limit}-file limit")
    return candidates[:limit]


def _changed_units(chunks: list[StoredChunk], info: PatchInfo) -> list[StoredChunk]:
    hit = []
    for c in chunks:
        n_added = sum(1 for n in range(c.start_line, c.end_line + 1) if n in info.added)
        if n_added:
            hit.append((n_added, c))
    hit.sort(key=lambda t: -t[0])
    chosen, size = [], 0
    for _n_added, c in hit:
        size += len(c.text)
        if size > MAX_UNITS_CHARS and chosen:
            break
        chosen.append(c)
    return sorted(chosen, key=lambda c: c.start_line)


def _units_text(units: list[StoredChunk], info: PatchInfo) -> str:
    parts = []
    for c in units:
        lines = c.text.splitlines()[:MAX_UNIT_LINES]
        body = number_lines("\n".join(lines), c.start_line, info.added)
        parts.append(f"# {c.qualname} (lines {c.start_line}-{c.end_line})\n{body}")
    return "\n\n".join(parts)


def _code_windows(text: str, comments: list[ReviewComment], radius: int = 25) -> str:
    lines = text.splitlines()
    spans: list[tuple[int, int]] = []
    for c in comments:
        lo = max(1, (c.start_line or c.line) - radius)
        hi = min(len(lines), c.line + radius)
        if spans and lo <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return "\n\n".join(number_lines("\n".join(lines[lo - 1 : hi]), lo) for lo, hi in spans)


def _finding_to_comment(f: Finding, file_text: str) -> ReviewComment:
    lines = file_text.splitlines()
    evidence = lines[f.line - 1].strip() if 0 < f.line <= len(lines) else ""
    category = "security" if f.rule.startswith(("S", "python.", "javascript.")) or "security" in f.rule else "bug"
    return ReviewComment(
        path=f.path, line=f.line, severity=f.severity, category=category, title=f.rule,
        explanation=f.message, evidence=evidence, confidence=0.6, source=f.source, finding_id=f.id,
    )


def run_review(
    inp: ReviewInput,
    llm: LLM | None,
    settings: Settings | None = None,
    opts: PipelineOptions | None = None,
    index_db: Path | None = None,
    triage: Callable[[dict], float] | None = None,
) -> ReviewResult:
    settings = settings or get_settings()
    opts = opts or PipelineOptions()
    cfg = inp.config
    result = ReviewResult()
    t0 = time.time()
    usage_before = llm.usage.snapshot() if llm is not None else Usage()

    files = _select_files(inp, settings, result.skipped)
    result.files_reviewed = len(files)
    if not files:
        result.seconds = time.time() - t0
        return result
    infos = {f.path: analyze_patch(f.patch) for f in files}
    patches = {f.path: f.patch for f in files}
    texts = {}
    for f in files:
        p = inp.repo_dir / f.path
        texts[f.path] = p.read_text(errors="replace") if p.is_file() else ""

    # --- index & retrieval -------------------------------------------------------------------
    index: RepoIndex | None = None
    retriever: Retriever | None = None
    if opts.use_context or opts.use_graph:
        db = index_db or (settings.index_dir / "default.db")
        index = RepoIndex(db, get_embedder(settings.embeddings_backend, settings.embedding_model))
        result.index_stats = index.update(inp.repo_dir, settings.max_file_bytes)
        retriever = Retriever(index)

    # --- deterministic findings --------------------------------------------------------------
    findings: list[Finding] = []
    if opts.use_static:
        raw, tool = analyze(inp.repo_dir, [f.path for f in files], settings.semgrep_config)
        result.static_tool = tool
        findings += only_on_lines(raw, {p: i.added for p, i in infos.items()})
    if opts.use_graph and index is not None:
        findings += detect_signature_changes(inp.repo_dir, inp.base_sha, inp.files, index)
    result.findings = findings

    conventions = load_conventions(inp.repo_dir, cfg) if opts.use_context else ""
    reviewer = Reviewer(llm) if (llm is not None and opts.llm_review) else None

    # --- per-file review ---------------------------------------------------------------------
    candidates: list[ReviewComment] = []
    for f in files:
        info = infos[f.path]
        file_findings = [x for x in findings if x.path == f.path]
        units_text = context_text = ""
        if index is not None and retriever is not None and opts.use_context:
            units = _changed_units(index.chunks_in_path(f.path), info)
            units_text = _units_text(units, info)
            items = retriever.expand(units)
            chosen, context_text, _used = build_context(items, settings.context_token_budget)
            result.context_paths.update(i.path for i in chosen)
        if reviewer is not None:
            comments, _dismissed = reviewer.review_file(
                path=f.path, patch=f.patch or "", units_text=units_text, context_text=context_text,
                findings=file_findings, conventions=conventions, title=inp.title, body=inp.body,
            )
            candidates.extend(comments)
        else:
            candidates.extend(
                _finding_to_comment(x, texts[f.path]) for x in file_findings if x.source == "static"
            )

    # --- verification ------------------------------------------------------------------------
    verifier = Verifier(texts, patches, infos, cfg, mode=opts.verify, triage=triage,
                        triage_threshold=settings.triage_threshold)
    do_llm_verify = cfg.llm_verify if opts.llm_verify is None else opts.llm_verify
    if opts.verify == "full" and do_llm_verify and reviewer is not None and candidates:
        # Cheap deterministic checks first, so the LLM only sees plausible comments.
        pre = verifier.check(candidates)
        kept: list[ReviewComment] = []
        for path in sorted({c.path for c in pre}):
            group = [c for c in pre if c.path == path]
            ok, dropped = reviewer.verify_comments(path, _code_windows(texts[path], group), group)
            kept.extend(ok)
            verifier.dropped.extend(dropped)
        final = verifier._rank(kept)
    else:
        final = verifier.check(candidates)

    result.comments = final
    result.dropped = verifier.dropped
    if index is not None:
        index.close()
    if llm is not None:
        after = llm.usage
        result.usage = Usage(
            after.input_tokens - usage_before.input_tokens,
            after.output_tokens - usage_before.output_tokens,
            after.calls - usage_before.calls,
            after.seconds - usage_before.seconds,
        )
    result.seconds = time.time() - t0
    return result
