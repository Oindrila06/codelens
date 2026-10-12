"""Embedding backends. The hashing embedder needs no model download and is deterministic."""
from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: `getUserById` -> getuserbyid, get, user, by, id."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        out.append(low)
        parts = [p.lower() for p in CAMEL_RE.findall(tok) if p != "_"]
        parts = [q for p in parts for q in p.split("_") if q]
        if len(parts) > 1:
            out.extend(parts)
    return out


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing of identifier tokens (a bag-of-words embedding)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = zlib.crc32(tok.encode())
                sign = 1.0 if (h >> 31) & 1 else -1.0
                mat[i, h % self.dim] += sign
            mat[i] = np.sign(mat[i]) * np.log1p(np.abs(mat[i]))
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(backend: str = "hash", model_name: str = "") -> Embedder:
    if backend in ("st", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:  # noqa: BLE001 - fall back rather than fail the review
            import logging

            logging.getLogger(__name__).warning("sentence-transformers unavailable; using hashing embedder")
    return HashingEmbedder()
