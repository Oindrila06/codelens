"""process_review end to end: real git fetch/checkout, real pipeline, fake GitHub and fake LLM."""
import json

from codelens.config import get_settings
from codelens.github_api import InstallationClient
from codelens.gitutil import GitError, diff_files, ensure_checkout, run_git
from codelens.jobs import process_review
from codelens.llm import FakeLLM
from codelens.storage import Storage
from codelens.testing import build_repo

from .test_pipeline import AFTER, BEFORE, scripted


class FakeGitHubClient:
    token = None

    def __init__(self, origin, base, head):
        self.origin, self.base, self.head = origin, base, head
        self.reviews = []

    def clone_url(self, owner, repo):
        return str(self.origin)

    def get_pr(self, owner, repo, number):
        return {"head": {"sha": self.head}, "base": {"sha": self.base}, "title": "Refactor", "body": ""}

    def list_pr_files(self, owner, repo, number):
        return diff_files(self.origin, self.base, self.head)

    def create_review(self, owner, repo, number, commit_id, body, comments):
        self.reviews.append((commit_id, body, comments))
        return {"id": 555}

    def list_review_comments(self, owner, repo, number, review_id):
        return [{"id": 1000 + i, "path": c.path, "line": c.line} for i, (_, _, cs) in enumerate(self.reviews)
                for c in cs]


class FakeApp:
    def __init__(self, client):
        self.client = client

    def client_for(self, installation_id):
        return self.client


def make(tmp_path):
    origin, base, head = build_repo(tmp_path / "origin", BEFORE, AFTER)
    run_git(["config", "uploadpack.allowAnySHA1InWant", "true"], origin)
    return origin, base, head


JOB = {"installation_id": 7, "owner": "o", "repo": "r", "pr_number": 3}


def test_ensure_checkout_fetches_shallow_and_checks_out_head(tmp_path):
    origin, base, head = make(tmp_path)
    dest = tmp_path / "clone"
    ensure_checkout(str(origin), dest, head, base)
    assert run_git(["rev-parse", "HEAD"], dest).strip() == head
    assert (dest / "run.py").read_text().count("shell=True") == 1
    ensure_checkout(str(origin), dest, base, None)  # re-use of an existing clone, moving to another commit
    assert run_git(["rev-parse", "HEAD"], dest).strip() == base
    try:
        ensure_checkout(str(origin), tmp_path / "c2", "0" * 40, None)
    except GitError:
        pass
    else:
        raise AssertionError("fetching an unknown sha must fail loudly")


def test_process_review_posts_comments_and_records_state(tmp_path, monkeypatch):
    monkeypatch.setenv("QUEUE_MODE", "inline")
    monkeypatch.setenv("SEMGREP_CONFIG", "p/python")
    get_settings.cache_clear()
    settings = get_settings()
    origin, base, head = make(tmp_path)
    client = FakeGitHubClient(origin, base, head)
    job = {**JOB, "head_sha": head}

    out = process_review(job, app=FakeApp(client), llm=FakeLLM(handler=scripted), settings=settings)
    assert out["status"] == "done" and out["comments"] >= 2
    commit_id, body, comments = client.reviews[0]
    assert commit_id == head and "CodeLens" in body
    assert {c.title for c in comments} >= {"Shell injection", "Breaking signature change"}

    st = Storage(settings.db_path)
    assert st.known_comment(1000) or st.known_comment(1001)
    assert st.summary()["reviews"] == 1
    payload = next(iter(st.comments_for_polling()))
    assert payload["repo"] == "o/r"

    # Same head sha again (webhook redelivery or a re-run): no second review.
    again = process_review(job, app=FakeApp(client), llm=FakeLLM(handler=scripted), settings=settings)
    assert again["status"] == "skipped" and len(client.reviews) == 1


def test_stale_head_is_skipped_and_failures_release_the_claim(tmp_path, monkeypatch):
    get_settings.cache_clear()
    settings = get_settings()
    origin, base, head = make(tmp_path)
    client = FakeGitHubClient(origin, base, head)
    out = process_review({**JOB, "head_sha": "f" * 40}, app=FakeApp(client), llm=FakeLLM(), settings=settings)
    assert out["status"] == "skipped" and "head moved" in out["reason"]

    class Exploding(FakeGitHubClient):
        def list_pr_files(self, *a, **k):
            raise RuntimeError("github 500")

    bad = Exploding(origin, base, head)
    try:
        process_review({**JOB, "head_sha": head}, app=FakeApp(bad), llm=FakeLLM(), settings=settings)
    except RuntimeError:
        pass
    # claim was released, so the retry can run and succeed
    ok = process_review({**JOB, "head_sha": head}, app=FakeApp(client), llm=FakeLLM(handler=scripted), settings=settings)
    assert ok["status"] == "done"


def test_clean_pr_posts_nothing(tmp_path):
    get_settings.cache_clear()
    settings = get_settings()
    origin, base, head = make(tmp_path)
    client = FakeGitHubClient(origin, base, head)
    out = process_review({**JOB, "head_sha": head}, app=FakeApp(client),
                         llm=FakeLLM(handler=lambda s, u: {"comments": []}), settings=settings)
    assert out["status"] == "done" and out["comments"] == 0 and client.reviews == []


def test_payload_is_json_serialisable():
    assert json.dumps(JOB)
    assert InstallationClient  # imported to guarantee the real class stays importable
