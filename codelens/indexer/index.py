"""Incremental repository index: chunks, references and embeddings in SQLite."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..gitutil import list_python_files
from .embeddings import Embedder
from .parser import parse_python

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL, name TEXT NOT NULL, qualname TEXT NOT NULL, kind TEXT NOT NULL,
  start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, text TEXT NOT NULL,
  signature TEXT NOT NULL DEFAULT '', is_test INTEGER NOT NULL DEFAULT 0,
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);
CREATE TABLE IF NOT EXISTS refs (
  chunk_id INTEGER NOT NULL, path TEXT NOT NULL, name TEXT NOT NULL, line INTEGER NOT NULL,
  kind TEXT NOT NULL, module TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(name, kind);
CREATE INDEX IF NOT EXISTS idx_refs_chunk ON refs(chunk_id);
CREATE INDEX IF NOT EXISTS idx_refs_path ON refs(path);
"""


@dataclass
class StoredChunk:
    id: int
    path: str
    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    text: str
    signature: str
    is_test: bool


def _row_to_chunk(r: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        id=r["id"], path=r["path"], name=r["name"], qualname=r["qualname"], kind=r["kind"],
        start_line=r["start_line"], end_line=r["end_line"], text=r["text"],
        signature=r["signature"], is_test=bool(r["is_test"]),
    )


class RepoIndex:
    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._matrix_cache: tuple[list[int], np.ndarray] | None = None
        self._check_embedder()

    def close(self) -> None:
        self.conn.close()

    def _check_embedder(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        if row and row["value"] != self.embedder.name:
            log.info("Embedder changed (%s -> %s); rebuilding index", row["value"], self.embedder.name)
            for t in ("files", "chunks", "refs"):
                self.conn.execute(f"DELETE FROM {t}")
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('embedder', ?)", (self.embedder.name,)
        )
        self.conn.commit()

    # ---- building -------------------------------------------------------------------------
    def update(self, repo_dir: Path, max_bytes: int = 300_000) -> dict[str, int]:
        repo_dir = Path(repo_dir)
        current: dict[str, str] = {}
        for rel in list_python_files(repo_dir, max_bytes):
            try:
                data = (repo_dir / rel).read_bytes()
            except OSError:
                continue
            if b"DO NOT EDIT" in data[:400] or b"auto-generated" in data[:400].lower():
                continue
            current[rel] = hashlib.sha1(data).hexdigest()
        known = {r["path"]: r["hash"] for r in self.conn.execute("SELECT path, hash FROM files")}
        removed = [p for p in known if p not in current]
        changed = [p for p, h in current.items() if known.get(p) != h]

        for path in removed:
            self._delete_file(path)
        for path in changed:
            source = (repo_dir / path).read_text(errors="replace")
            self._index_file(path, source)
            self.conn.execute(
                "INSERT OR REPLACE INTO files(path, hash) VALUES(?, ?)", (path, current[path])
            )
        self.conn.commit()
        self._matrix_cache = None
        return {"indexed": len(changed), "unchanged": len(current) - len(changed), "removed": len(removed)}

    def _delete_file(self, path: str) -> None:
        self.conn.execute("DELETE FROM refs WHERE path=?", (path,))
        self.conn.execute("DELETE FROM chunks WHERE path=?", (path,))
        self.conn.execute("DELETE FROM files WHERE path=?", (path,))

    def _index_file(self, path: str, source: str) -> None:
        self.conn.execute("DELETE FROM refs WHERE path=?", (path,))
        self.conn.execute("DELETE FROM chunks WHERE path=?", (path,))
        try:
            parsed = parse_python(path, source)
        except Exception as exc:  # noqa: BLE001 - a bad file must not break indexing
            log.warning("Could not parse %s: %s", path, exc)
            return
        if not parsed.chunks:
            return
        texts = [f"{c.qualname}\n{c.text[:1500]}" for c in parsed.chunks]
        vectors = self.embedder.embed(texts)
        ids: list[int] = []
        for chunk, vec in zip(parsed.chunks, vectors, strict=True):
            cur = self.conn.execute(
                "INSERT INTO chunks(path,name,qualname,kind,start_line,end_line,text,signature,is_test,embedding)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (path, chunk.name, chunk.qualname, chunk.kind, chunk.start_line, chunk.end_line,
                 chunk.text, chunk.signature, int(chunk.is_test), vec.astype(np.float32).tobytes()),
            )
            ids.append(int(cur.lastrowid))
        self.conn.executemany(
            "INSERT INTO refs(chunk_id,path,name,line,kind,module) VALUES(?,?,?,?,?,?)",
            [(ids[r.chunk_index], path, r.name, r.line, r.kind, r.module) for r in parsed.refs],
        )

    # ---- queries --------------------------------------------------------------------------
    def chunks_in_path(self, path: str) -> list[StoredChunk]:
        rows = self.conn.execute("SELECT * FROM chunks WHERE path=? ORDER BY start_line", (path,))
        return [_row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: int) -> StoredChunk | None:
        r = self.conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return _row_to_chunk(r) if r else None

    def chunks_named(self, name: str, kinds: tuple[str, ...] = ("function", "class")) -> list[StoredChunk]:
        q = ",".join("?" * len(kinds))
        rows = self.conn.execute(f"SELECT * FROM chunks WHERE name=? AND kind IN ({q})", (name, *kinds))
        return [_row_to_chunk(r) for r in rows]

    def refs_named(self, name: str, kind: str = "call") -> list[tuple[int, str, int]]:
        rows = self.conn.execute(
            "SELECT chunk_id, path, line FROM refs WHERE name=? AND kind=?", (name, kind)
        )
        return [(r["chunk_id"], r["path"], r["line"]) for r in rows]

    def calls_from(self, chunk_id: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT name FROM refs WHERE chunk_id=? AND kind='call'", (chunk_id,)
        )
        return [r["name"] for r in rows]

    def imports_of(self, path: str) -> set[tuple[str, str]]:
        rows = self.conn.execute("SELECT name, module FROM refs WHERE path=? AND kind='import'", (path,))
        return {(r["name"], r["module"]) for r in rows}

    def all_chunks(self) -> list[StoredChunk]:
        return [_row_to_chunk(r) for r in self.conn.execute("SELECT * FROM chunks")]

    def embedding_matrix(self) -> tuple[list[int], np.ndarray]:
        if self._matrix_cache is None:
            rows = self.conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
            ids = [r["id"] for r in rows]
            if rows:
                mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            else:
                mat = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._matrix_cache = (ids, mat)
        return self._matrix_cache
