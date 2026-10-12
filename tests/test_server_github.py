import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from codelens import jobs, server
from codelens.config import get_settings
from codelens.github_api import GitHubError, InstallationClient, render_comment_body
from codelens.schemas import ReviewComment
from codelens.storage import Storage


def signed(body: dict, secret="testsecret"):
    raw = json.dumps(body).encode()
    return raw, "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def pr_payload(action="opened", draft=False):
    return {
        "action": action,
        "installation": {"id": 7},
        "repository": {"name": "r", "owner": {"login": "o"}},
        "pull_request": {"number": 3, "draft": draft, "head": {"sha": "abc123"}},
    }


@pytest.fixture
def client(monkeypatch):
    queued = []
    monkeypatch.setattr(server, "process_review_safe", lambda job: queued.append(job))
    c = TestClient(server.app)
    c.queued = queued
    return c


def post(client, body, event="pull_request", delivery="d1", secret="testsecret"):
    raw, sig = signed(body, secret)
    return client.post("/webhook", content=raw, headers={
        "X-GitHub-Event": event, "X-GitHub-Delivery": delivery, "X-Hub-Signature-256": sig,
        "Content-Type": "application/json"})


def test_rejects_bad_signature(client):
    assert post(client, pr_payload(), secret="wrong").status_code == 401
    assert client.queued == []


def test_queues_pull_request_once(client):
    r = post(client, pr_payload())
    assert r.json()["status"] == "queued" and len(client.queued) == 1
    assert client.queued[0] == {"installation_id": 7, "owner": "o", "repo": "r", "pr_number": 3, "head_sha": "abc123"}
    # GitHub retries the same delivery id: must not queue again.
    assert post(client, pr_payload()).json()["status"] == "duplicate"
    assert len(client.queued) == 1


def test_ignores_drafts_closed_and_other_events(client):
    assert post(client, pr_payload(draft=True), delivery="a").json()["status"] == "ignored"
    assert post(client, pr_payload(action="closed"), delivery="b").json()["status"] == "ignored"
    assert post(client, {}, event="push", delivery="c").json()["status"] == "ignored"
    assert post(client, {}, event="ping", delivery="d").json()["status"] == "pong"
    assert client.queued == []


def test_thread_resolution_recorded_as_feedback(client):
    st = Storage(get_settings().db_path)
    st.record_comment(99, "o/r", 3, "abc", 7, {"path": "a.py", "line": 1, "severity": "high", "category": "bug",
                                              "title": "t", "source": "llm"})
    body = {"action": "resolved", "thread": {"comments": [{"id": 99}]}}
    assert post(client, body, event="pull_request_review_thread", delivery="x").json()["signal"] == "resolved"
    assert any(row["signals"] == ["resolved"] for row in st.labeled_comments())
    body["action"] = "unresolved"
    post(client, body, event="pull_request_review_thread", delivery="y")
    assert st.labeled_comments() == []
    other = {"action": "resolved", "thread": {"comments": [{"id": 12345}]}}
    assert post(client, other, event="pull_request_review_thread", delivery="z").json()["status"] == "ignored"


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_review_state_is_idempotent(tmp_path):
    st = Storage(tmp_path / "s.db")
    assert st.start_review("o/r", 1, "sha", 7)
    assert not st.start_review("o/r", 1, "sha", 7)  # already running
    st.abort_review("o/r", 1, "sha")
    assert st.start_review("o/r", 1, "sha", 7)  # retry allowed after failure
    st.finish_review("o/r", 1, "sha", 5, 2, 100, 50, 0.01, 3.0)
    assert not st.start_review("o/r", 1, "sha", 7)
    assert st.summary()["comments_posted"] == 2


def test_comment_body_rendering():
    c = ReviewComment(path="a.py", line=3, severity="high", category="security", title="Injection",
                      explanation="Because `x` is unsanitised.", suggestion="    y = 1", confidence=0.8,
                      related_files=["b.py"])
    body = render_comment_body(c)
    assert "HIGH · security" in body and "```suggestion\n    y = 1\n```" in body and "`b.py`" in body
    c.suggestion = "x = '''```'''"
    assert "````suggestion" in render_comment_body(c)  # longer fence when suggestion contains ```


def test_comment_payload_multiline():
    c = ReviewComment(path="a.py", line=5, start_line=3, title="t", explanation="e")
    p = InstallationClient._comment_payload(c)
    assert p["line"] == 5 and p["start_line"] == 3 and p["start_side"] == "RIGHT" and p["side"] == "RIGHT"
    single = InstallationClient._comment_payload(ReviewComment(path="a.py", line=5))
    assert "start_line" not in single


class FakeClient:
    def __init__(self, reject_batch):
        self.reject_batch, self.single = reject_batch, []

    def create_review(self, *a, **k):
        if self.reject_batch:
            raise GitHubError(422, "line must be part of the diff")
        return {"id": 11}

    def list_review_comments(self, *a, **k):
        return [{"id": 1, "path": "a.py", "line": 3}]

    def create_single_comment(self, owner, repo, number, sha, c):
        if c.line == 99:
            raise GitHubError(422, "bad anchor")
        self.single.append(c.line)
        return {"id": 100 + c.line, "path": c.path, "line": c.line}


def test_post_falls_back_to_individual_comments_on_422():
    comments = [ReviewComment(path="a.py", line=3), ReviewComment(path="a.py", line=99)]
    review_id, posted = jobs._post(FakeClient(True), "o", "r", 1, "sha", "body", comments)
    assert review_id is None and [p["line"] for p in posted] == [3]
    review_id, posted = jobs._post(FakeClient(False), "o", "r", 1, "sha", "body", comments)
    assert review_id == 11 and len(posted) == 1


def test_non_422_errors_propagate():
    class Boom(FakeClient):
        def create_review(self, *a, **k):
            raise GitHubError(500, "server error")

    with pytest.raises(GitHubError):
        jobs._post(Boom(False), "o", "r", 1, "sha", "b", [ReviewComment(path="a.py", line=1)])
