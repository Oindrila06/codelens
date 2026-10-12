"""Fit retrieved context into a token budget, highest-priority items first."""
from __future__ import annotations

from .retrieval import ContextItem

MAX_ITEM_LINES = 60


def estimate_tokens(text: str) -> int:
    """Rough estimate (about 4 characters per token); good enough for budgeting."""
    return len(text) // 4 + 1


def render_item(item: ContextItem) -> str:
    lines = item.text.splitlines()
    truncated = len(lines) > MAX_ITEM_LINES
    body = "\n".join(lines[:MAX_ITEM_LINES]) + ("\n# ... (truncated)" if truncated else "")
    header = f"### {item.path}:{item.start_line}-{item.end_line} `{item.qualname}`  [{item.reason}]"
    return f"{header}\n```python\n{body}\n```"


def build_context(items: list[ContextItem], token_budget: int) -> tuple[list[ContextItem], str, int]:
    selected: list[ContextItem] = []
    parts: list[str] = []
    used = 0
    for item in items:
        rendered = render_item(item)
        cost = estimate_tokens(rendered)
        if used + cost > token_budget:
            continue
        selected.append(item)
        parts.append(rendered)
        used += cost
    return selected, "\n\n".join(parts), used
