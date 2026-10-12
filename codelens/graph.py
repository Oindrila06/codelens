"""Deterministic cross-file checks driven by the code graph."""
from __future__ import annotations

from pathlib import Path

from .gitutil import read_file_at
from .indexer.index import RepoIndex
from .indexer.parser import parse_python, signatures
from .schemas import ChangedFile, Finding


def _call_sites(index: RepoIndex, name: str, exclude_paths: set[str], limit: int = 6) -> list[str]:
    sites: list[str] = []
    for _chunk_id, path, line in index.refs_named(name, "call"):
        if path in exclude_paths:
            continue
        sites.append(f"{path}:{line}")
    return sorted(set(sites))[:limit]


def detect_signature_changes(
    repo_dir: Path, base_sha: str, files: list[ChangedFile], index: RepoIndex
) -> list[Finding]:
    """Flag functions whose signature changed (or that were removed) while code outside the PR
    still calls them. This is the classic cross-file bug that a diff-only reviewer cannot see."""
    changed_paths = {f.path for f in files}
    findings: list[Finding] = []
    n = 0
    for f in files:
        if not f.path.endswith(".py") or f.status in ("added", "removed"):
            continue
        old_src = read_file_at(repo_dir, base_sha, f.path)
        new_path = repo_dir / f.path
        if old_src is None or not new_path.is_file():
            continue
        old = signatures(parse_python(f.path, old_src))
        new_parsed = parse_python(f.path, new_path.read_text(errors="replace"))
        new = signatures(new_parsed)
        start_of = {c.qualname: c.start_line for c in new_parsed.chunks}
        for qual, old_sig in old.items():
            name = qual.split(".")[-1]
            if name.startswith("__") and name.endswith("__"):
                continue
            if qual in new and new[qual] != old_sig:
                sites = _call_sites(index, name, exclude_paths=changed_paths)
                if sites:
                    n += 1
                    line = start_of.get(qual, 1)
                    findings.append(
                        Finding(
                            id=f"gr{n}", path=f.path, line=line, end_line=line,
                            rule="signature-changed",
                            message=(
                                f"Signature of `{qual}` changed from `{old_sig}` to `{new[qual]}`. "
                                f"These call sites are outside this PR and may need updating: "
                                f"{', '.join(sites)}"
                            ),
                            severity="high", source="graph", extra={"call_sites": sites},
                        )
                    )
            elif qual not in new:
                sites = _call_sites(index, name, exclude_paths=changed_paths)
                if sites:
                    n += 1
                    findings.append(
                        Finding(
                            id=f"gr{n}", path=f.path, line=1, end_line=1, rule="function-removed",
                            message=(
                                f"`{qual}` was removed or renamed but is still referenced at: "
                                f"{', '.join(sites)}"
                            ),
                            severity="high", source="graph", extra={"call_sites": sites},
                        )
                    )
    return findings
