"""Background work: run a review for one pull request and post the result to GitHub."""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import Settings, get_settings
from .github_api import (  # noqa: F401
    GitHubApp,
    GitHubError,
    InstallationClient,
    render_comment_body,
)
from .gitutil import ensure_checkout
from .llm import AnthropicLLM
from .pipeline import PipelineOptions, run_review
from .repo_config import load_repo_config
from .schemas import ReviewComment, ReviewInput
from .storage import Storage
from .triage import load_triage

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _triage(model_dir: str):
    return load_triage(model_dir)


def _summary_body(n_files: int, comments: list[ReviewComment], skipped: list[str], cost_line: str) -> str:
    sev: dict[str, int] = {}
    for c in comments:
        sev[c.severity] = sev.get(c.severity, 0) + 1
    breakdown = ", ".join(f"{n} {s}" for s, n in sorted(sev.items(), key=lambda kv: -kv[1]))
    lines = [f"**CodeLens** reviewed {n_files} Python file(s) and left {len(comments)} comment(s)"
             + (f" ({breakdown})." if breakdown else ".")]
    if skipped:
        lines.append("<details><summary>Skipped</summary>\n\n" + "\n".join(f"- {s}" for s in skipped[:15]) + "\n\n</details>")
    if cost_line:
        lines.append(f"<sub>{cost_line}</sub>")
    return "\n\n".join(lines)


def _post(client: InstallationClient, owner: str, repo: str, number: int, sha: str, body: str,
          comments: list[ReviewComment]) -> tuple[int | None, list[dict]]:
    """Post as one review; if GitHub rejects an anchor (422), fall back to one comment at a time."""
    try:
        review = client.create_review(owner, repo, number, sha, body, comments)
        posted = client.list_review_comments(owner, repo, number, review["id"])
        return review["id"], posted
    except GitHubError as exc:
        if exc.status != 422:
            raise
        log.warning("Batch review rejected (%s); posting comments individually", exc)
    posted, failed = [], 0
    for c in comments:
        try:
            posted.append(client.create_single_comment(owner, repo, number, sha, c))
        except GitHubError as exc:
            failed += 1
            log.warning("Could not post comment on %s:%s (%s)", c.path, c.line, exc)
    if not posted:
        return None, []
    return None, posted


def process_review(job: dict, app: GitHubApp | None = None, llm=None, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    storage = Storage(settings.db_path)
    owner, repo, number = job["owner"], job["repo"], int(job["pr_number"])
    full_name, head_sha = f"{owner}/{repo}", job["head_sha"]
    installation_id = int(job["installation_id"])

    if not storage.start_review(full_name, number, head_sha, installation_id):
        return {"status": "skipped", "reason": "already reviewed or in progress"}
    try:
        app = app or GitHubApp(settings)
        client = app.client_for(installation_id)

        pr = client.get_pr(owner, repo, number)
        if pr["head"]["sha"] != head_sha:
            storage.abort_review(full_name, number, head_sha)
            return {"status": "skipped", "reason": "head moved; a newer review is queued"}
        base_sha = pr["base"]["sha"]

        dest = settings.repos_dir / f"{owner}__{repo}"
        ensure_checkout(client.clone_url(owner, repo), dest, head_sha, base_sha, number, client.token)
        cfg = load_repo_config(dest)
        if not cfg.enabled:
            storage.finish_review(full_name, number, head_sha, None, 0, 0, 0, 0.0, 0.0)
            return {"status": "disabled"}

        files = client.list_pr_files(owner, repo, number)
        inp = ReviewInput(repo_dir=dest, base_sha=base_sha, head_sha=head_sha, files=files,
                          title=pr.get("title") or "", body=pr.get("body") or "", config=cfg)
        llm = llm or AnthropicLLM(settings)
        triage = _triage(settings.triage_model_dir)
        result = run_review(
            inp, llm, settings, PipelineOptions(),
            index_db=settings.index_dir / f"{owner}__{repo}.db",
            triage=triage.predict_proba if triage else None,
        )

        review_id, posted = None, []
        if result.comments:
            cost = result.usage.cost(settings.price_in_per_mtok, settings.price_out_per_mtok)
            cost_line = f"{result.usage.input_tokens + result.usage.output_tokens} tokens · {result.seconds:.0f}s"
            if cost:
                cost_line += f" · ~${cost:.3f}"
            body = _summary_body(result.files_reviewed, result.comments, result.skipped, cost_line)
            review_id, posted = _post(client, owner, repo, number, head_sha, body, result.comments)

        # Record posted comments (matched by path+line) so reactions and resolutions can be tracked.
        by_anchor = {(c.path, c.line): c for c in result.comments}
        for gh in posted:
            c = by_anchor.get((gh.get("path"), gh.get("line") or gh.get("original_line")))
            if c is not None:
                storage.record_comment(int(gh["id"]), full_name, number, head_sha, installation_id, c.model_dump())

        storage.finish_review(
            full_name, number, head_sha, review_id, len(posted), result.usage.input_tokens,
            result.usage.output_tokens,
            result.usage.cost(settings.price_in_per_mtok, settings.price_out_per_mtok), result.seconds,
        )
        return {"status": "done", "comments": len(posted), "files": result.files_reviewed,
                "dropped": len(result.dropped)}
    except Exception:
        storage.abort_review(full_name, number, head_sha)
        raise


def process_review_safe(job: dict) -> None:
    """Used in inline mode, where there is no queue to retry or report failures."""
    try:
        log.info("review finished: %s", process_review(job))
    except Exception:  # noqa: BLE001
        log.exception("review failed for %s", job)


def enqueue(job: dict, settings: Settings | None = None) -> None:
    """Redis/RQ mode. (Inline mode is handled by the web layer with BackgroundTasks.)"""
    from redis import Redis
    from rq import Queue, Retry

    settings = settings or get_settings()
    Queue("reviews", connection=Redis.from_url(settings.redis_url)).enqueue(
        "codelens.jobs.process_review", job, job_timeout=1800, retry=Retry(max=2, interval=[30, 120]),
        result_ttl=3600, failure_ttl=86400,
    )
