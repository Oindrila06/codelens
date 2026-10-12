"""Unified-diff helpers: which new-file lines were added, and which are commentable."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class PatchInfo:
    added: set[int] = field(default_factory=set)  # new-file line numbers that were added
    commentable: set[int] = field(default_factory=set)  # added + context lines (RIGHT side)
    removed_old: set[int] = field(default_factory=set)
    hunks: list[tuple[int, int]] = field(default_factory=list)  # (new_start, new_end)


def analyze_patch(patch: str | None) -> PatchInfo:
    """Parse the `patch` text GitHub returns for a file (starts at the first @@)."""
    info = PatchInfo()
    if not patch:
        return info
    new_no = old_no = 0
    hunk_start = hunk_end = 0
    in_hunk = False
    for raw in patch.splitlines():
        m = HUNK_RE.match(raw)
        if m:
            if in_hunk:
                info.hunks.append((hunk_start, max(hunk_end, hunk_start)))
            old_no = int(m.group(1))
            new_no = int(m.group(3))
            hunk_start = new_no
            hunk_end = new_no
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        if raw.startswith("+"):
            info.added.add(new_no)
            info.commentable.add(new_no)
            hunk_end = new_no
            new_no += 1
        elif raw.startswith("-"):
            info.removed_old.add(old_no)
            old_no += 1
        else:  # context line (may be a bare empty string if trailing space was stripped)
            info.commentable.add(new_no)
            hunk_end = new_no
            new_no += 1
            old_no += 1
    if in_hunk:
        info.hunks.append((hunk_start, max(hunk_end, hunk_start)))
    return info


def contiguous_ranges(lines: set[int]) -> list[tuple[int, int]]:
    """[3,4,5,9] -> [(3,5),(9,9)]"""
    out: list[tuple[int, int]] = []
    for n in sorted(lines):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def nearest_commentable(line: int, commentable: set[int], max_dist: int = 3) -> int | None:
    """Snap a line number to the closest line that exists in the diff."""
    if line in commentable:
        return line
    best: int | None = None
    for d in range(1, max_dist + 1):
        for cand in (line + d, line - d):
            if cand in commentable:
                best = cand
                break
        if best is not None:
            break
    return best


def same_hunk(a: int, b: int, hunks: list[tuple[int, int]]) -> bool:
    return any(s <= a <= e and s <= b <= e for s, e in hunks)
