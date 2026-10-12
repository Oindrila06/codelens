"""SQLite storage: webhook idempotency, review records, posted comments and feedback signals."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (id TEXT PRIMARY KEY, received_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
  repo TEXT NOT NULL, pr INTEGER NOT NULL, head_sha TEXT NOT NULL,
  installation_id INTEGER, review_id INTEGER, status TEXT NOT NULL,
  started_at REAL NOT NULL, finished_at REAL, comments_posted INTEGER DEFAULT 0,
  tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost REAL DEFAULT 0, seconds REAL DEFAULT 0,
  PRIMARY KEY (repo, pr, head_sha)
);
CREATE TABLE IF NOT EXISTS comments (
  gh_comment_id INTEGER PRIMARY KEY, repo TEXT NOT NULL, pr INTEGER NOT NULL, head_sha TEXT NOT NULL,
  installation_id INTEGER, path TEXT, line INTEGER, severity TEXT, category TEXT, title TEXT,
  source TEXT, payload TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, gh_comment_id INTEGER NOT NULL, signal TEXT NOT NULL,
  created_at REAL NOT NULL, UNIQUE (gh_comment_id, signal)
);
"""


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- idempotency ---------------------------------------------------------------------------
    def claim_delivery(self, delivery_id: str) -> bool:
        """True the first time a delivery id is seen; False for retries and duplicates."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO deliveries(id, received_at) VALUES(?, ?)", (delivery_id, time.time())
            )
            return cur.rowcount == 1

    def start_review(self, repo: str, pr: int, head_sha: str, installation_id: int) -> bool:
        """Claim (repo, pr, head_sha). False if it is already running or done."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO reviews(repo, pr, head_sha, installation_id, status, started_at)"
                " VALUES(?,?,?,?, 'running', ?)",
                (repo, pr, head_sha, installation_id, time.time()),
            )
            return cur.rowcount == 1

    def abort_review(self, repo: str, pr: int, head_sha: str) -> None:
        """Release the claim after a failure so a retry can run."""
        with self._conn() as c:
            c.execute("DELETE FROM reviews WHERE repo=? AND pr=? AND head_sha=? AND status='running'",
                      (repo, pr, head_sha))

    def finish_review(self, repo: str, pr: int, head_sha: str, review_id: int | None, posted: int,
                      tokens_in: int, tokens_out: int, cost: float, seconds: float) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE reviews SET status='done', review_id=?, finished_at=?, comments_posted=?,"
                " tokens_in=?, tokens_out=?, cost=?, seconds=? WHERE repo=? AND pr=? AND head_sha=?",
                (review_id, time.time(), posted, tokens_in, tokens_out, cost, seconds, repo, pr, head_sha),
            )

    # --- comments & feedback ---------------------------------------------------------------------
    def record_comment(self, gh_comment_id: int, repo: str, pr: int, head_sha: str,
                       installation_id: int, payload: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO comments(gh_comment_id, repo, pr, head_sha, installation_id, path, line,"
                " severity, category, title, source, payload, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (gh_comment_id, repo, pr, head_sha, installation_id, payload.get("path"), payload.get("line"),
                 payload.get("severity"), payload.get("category"), payload.get("title"),
                 payload.get("source"), json.dumps(payload), time.time()),
            )

    def record_feedback(self, gh_comment_id: int, signal: str) -> bool:
        """Signals: thumbs_up, thumbs_down, resolved, unresolved, replied. Returns True if new."""
        with self._conn() as c:
            if signal == "unresolved":
                c.execute("DELETE FROM feedback WHERE gh_comment_id=? AND signal='resolved'", (gh_comment_id,))
                return True
            cur = c.execute(
                "INSERT OR IGNORE INTO feedback(gh_comment_id, signal, created_at) VALUES(?,?,?)",
                (gh_comment_id, signal, time.time()),
            )
            return cur.rowcount == 1

    def known_comment(self, gh_comment_id: int) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM comments WHERE gh_comment_id=?", (gh_comment_id,)).fetchone() is not None

    def comments_for_polling(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT gh_comment_id, repo, installation_id FROM comments")]

    def labeled_comments(self) -> list[dict]:
        """Comments with at least one feedback signal, for fine-tuning data."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT c.gh_comment_id, c.repo, c.payload, GROUP_CONCAT(f.signal) AS signals "
                "FROM comments c JOIN feedback f ON f.gh_comment_id = c.gh_comment_id GROUP BY c.gh_comment_id"
            ).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"]), "signals": r["signals"].split(",")} for r in rows]

    def summary(self) -> dict:
        with self._conn() as c:
            reviews = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(comments_posted),0) posted, COALESCE(SUM(tokens_in),0) tin,"
                " COALESCE(SUM(tokens_out),0) tout, COALESCE(SUM(cost),0) cost, COALESCE(AVG(seconds),0) avg_s"
                " FROM reviews WHERE status='done'"
            ).fetchone()
            sig = {r["signal"]: r["n"] for r in c.execute("SELECT signal, COUNT(*) n FROM feedback GROUP BY signal")}
            total_comments = c.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        return {"reviews": reviews["n"], "comments_posted": reviews["posted"], "tracked_comments": total_comments,
                "tokens_in": reviews["tin"], "tokens_out": reviews["tout"], "cost": reviews["cost"],
                "avg_seconds": reviews["avg_s"], "feedback": sig}
