"""Hybrid retrieval (BM25 + embeddings) and code-graph expansion around changed code."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np

from .embeddings import tokenize
from .index import RepoIndex, StoredChunk

# Reasons, ordered by how useful they usually are for understanding a change.
REASON_PRIORITY = {"caller": 0, "callee": 1, "class": 2, "test": 3, "similar": 4}
COMMON_NAMES = {
    "get", "set", "add", "run", "main", "init", "open", "close", "read", "write", "load", "dump",
    "update", "delete", "items", "keys", "values", "append", "format", "join", "split", "strip",
    "print", "len", "str", "int", "list", "dict", "super", "isinstance", "range", "map", "filter",
}


@dataclass
class ContextItem:
    chunk_id: int
    path: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    text: str
    reason: str
    score: float = 0.0
    category: str = "similar"  # caller | callee | class | test | similar

    @property
    def priority(self) -> int:
        return REASON_PRIORITY.get(self.category, 9)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in docs]
        self.avg = (sum(self.doc_len) / len(docs)) if docs else 0.0
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            counts: dict[str, int] = defaultdict(int)
            for tok in doc:
                counts[tok] += 1
            for tok, tf in counts.items():
                self.inverted[tok].append((i, tf))
        self.n = len(docs)

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for tok in set(query):
            postings = self.inverted.get(tok)
            if not postings:
                continue
            idf = math.log(1 + (self.n - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / (self.avg or 1))
                out[doc_id] += idf * tf * (self.k1 + 1) / denom
        return out


class Retriever:
    def __init__(self, index: RepoIndex) -> None:
        self.index = index
        self._bm25: BM25 | None = None
        self._ids: list[int] = []
        self._build()

    def _build(self) -> None:
        chunks = self.index.all_chunks()
        self._ids = [c.id for c in chunks]
        self._bm25 = BM25([tokenize(f"{c.qualname} {c.text}") for c in chunks])

    # ---- search ---------------------------------------------------------------------------
    def search(self, query: str, k: int = 5, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """Reciprocal-rank fusion of BM25 and embedding similarity."""
        exclude = exclude or set()
        if not self._ids:
            return []
        bm = self._bm25.scores(tokenize(query)) if self._bm25 else {}
        bm_rank = {self._ids[i]: r for r, (i, _) in enumerate(sorted(bm.items(), key=lambda kv: -kv[1])[:50])}
        ids, mat = self.index.embedding_matrix()
        emb_rank: dict[int, int] = {}
        if len(ids):
            qv = self.index.embedder.embed([query])[0]
            sims = mat @ qv
            order = np.argsort(-sims)[:50]
            emb_rank = {ids[int(i)]: r for r, i in enumerate(order) if sims[int(i)] > 0.05}
        fused: dict[int, float] = defaultdict(float)
        for ranks in (bm_rank, emb_rank):
            for cid, r in ranks.items():
                fused[cid] += 1.0 / (60 + r)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return [(cid, s) for cid, s in ranked if cid not in exclude][:k]

    # ---- graph expansion ------------------------------------------------------------------
    def _proximity(self, a: str, b: str) -> int:
        pa, pb = PurePosixPath(a), PurePosixPath(b)
        if pa.parent == pb.parent:
            return 2
        return 1 if pa.parts[:1] == pb.parts[:1] else 0

    def _imports_target(self, caller_path: str, target: StoredChunk) -> bool:
        stem = PurePosixPath(target.path).stem
        for name, module in self.index.imports_of(caller_path):
            if name == target.name or module.split(".")[-1] == stem or name == stem:
                return True
        return False

    def expand(
        self,
        changed: list[StoredChunk],
        per_chunk_limit: int = 6,
        exclude_paths: set[str] | None = None,
    ) -> list[ContextItem]:
        """Gather the code a reviewer would open to understand `changed`."""
        exclude_paths = exclude_paths or set()
        changed_ids = {c.id for c in changed}
        items: dict[int, ContextItem] = {}

        def add(chunk: StoredChunk | None, category: str, reason: str, score: float) -> None:
            if chunk is None or chunk.id in changed_ids or chunk.path in exclude_paths:
                return
            existing = items.get(chunk.id)
            item = ContextItem(chunk.id, chunk.path, chunk.qualname, chunk.kind, chunk.start_line,
                               chunk.end_line, chunk.text, reason, score, category)
            if existing is None or item.priority < existing.priority:
                items[chunk.id] = item

        for ch in changed:
            before = set(items)

            # Callers (and tests that exercise the changed code)
            if ch.kind in ("function", "class") and ch.name not in COMMON_NAMES and len(ch.name) >= 3:
                callers = []
                for chunk_id, _path, _line in self.index.refs_named(ch.name, "call"):
                    caller = self.index.get_chunk(chunk_id)
                    if caller is None or caller.id == ch.id:
                        continue
                    score = self._proximity(caller.path, ch.path) + (3 if self._imports_target(caller.path, ch) else 0)
                    callers.append((score, caller))
                callers.sort(key=lambda t: -t[0])
                for score, caller in callers[:per_chunk_limit]:
                    if caller.is_test:
                        add(caller, "test", f"test of `{ch.qualname}`", float(score))
                    else:
                        add(caller, "caller", f"caller of `{ch.qualname}`", float(score))

            # Callees: definitions of functions the changed code invokes
            for name in self.index.calls_from(ch.id):
                if name in COMMON_NAMES or len(name) < 3:
                    continue
                defs = self.index.chunks_named(name)
                defs.sort(key=lambda d: -self._proximity(d.path, ch.path))
                for d in defs[:2]:
                    add(d, "callee", f"callee used by `{ch.qualname}`", 1.0 + self._proximity(d.path, ch.path))

            # Enclosing class header for methods
            if "." in ch.qualname:
                cls_name = ch.qualname.split(".")[-2]
                for d in self.index.chunks_in_path(ch.path):
                    if d.kind == "class" and d.name == cls_name:
                        add(d, "class", "enclosing class", 1.0)

            # Semantically similar code (conventions, duplicate logic)
            if ch.kind in ("function", "class"):
                query = f"{ch.qualname} {ch.signature} " + " ".join(ch.text.splitlines()[:12])
                for cid, score in self.search(query, k=2, exclude=changed_ids):
                    add(self.index.get_chunk(cid), "similar", "similar code", score)

            new_ids = [i for i in items if i not in before]
            if len(new_ids) > per_chunk_limit:
                keep = sorted(new_ids, key=lambda i: (items[i].priority, -items[i].score))[:per_chunk_limit]
                for i in new_ids:
                    if i not in keep:
                        items.pop(i)
        return sorted(items.values(), key=lambda it: (it.priority, -it.score))
