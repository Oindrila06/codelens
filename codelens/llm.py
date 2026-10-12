"""LLM access: a small interface, an Anthropic implementation, and robust JSON extraction."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0

    def cost(self, price_in: float, price_out: float) -> float:
        return self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out

    def snapshot(self) -> Usage:
        return Usage(self.input_tokens, self.output_tokens, self.calls, self.seconds)


class LLM(Protocol):
    usage: Usage

    def complete_json(self, system: str, user: str, max_tokens: int | None = None) -> dict: ...


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates fences and chatter)."""
    candidates = [text]
    candidates += _FENCE_RE.findall(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in model output")


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.default_max_tokens = settings.llm_max_tokens
        self.usage = Usage()

    def _call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages
        )
        self.usage.seconds += time.time() - t0
        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def complete_json(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        max_tokens = max_tokens or self.default_max_tokens
        messages = [{"role": "user", "content": user}]
        text = self._call(system, messages, max_tokens)
        try:
            return extract_json(text)
        except ValueError:
            log.warning("Model returned invalid JSON; retrying once")
            messages += [
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
            ]
            return extract_json(self._call(system, messages, max_tokens))


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests and offline demos. `handler(system, user) -> dict`."""

    handler: object = None
    usage: Usage = field(default_factory=Usage)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete_json(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        self.calls.append((system, user))
        self.usage.calls += 1
        self.usage.input_tokens += (len(system) + len(user)) // 4
        out = self.handler(system, user) if callable(self.handler) else {"comments": []}
        self.usage.output_tokens += len(json.dumps(out)) // 4
        return out
