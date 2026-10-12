"""FastAPI webhook receiver: verify, deduplicate, enqueue, return fast."""
from __future__ import annotations

import json
import logging

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from .config import get_settings
from .jobs import enqueue, process_review_safe
from .security import verify_signature
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("codelens.server")

app = FastAPI(title="CodeLens", version="0.1.0")
REVIEW_ACTIONS = {"opened", "reopened", "synchronize", "ready_for_review"}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


def build_job(payload: dict) -> dict:
    pr, repo = payload["pull_request"], payload["repository"]
    return {
        "installation_id": payload["installation"]["id"],
        "owner": repo["owner"]["login"],
        "repo": repo["name"],
        "pr_number": pr["number"],
        "head_sha": pr["head"]["sha"],
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    background: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    settings = get_settings()
    body = await request.body()
    if not verify_signature(settings.github_webhook_secret, body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="invalid signature")
    if not x_github_delivery:
        raise HTTPException(status_code=400, detail="missing delivery id")

    storage = Storage(settings.db_path)
    if not storage.claim_delivery(x_github_delivery):
        return {"status": "duplicate"}

    if x_github_event == "ping":
        return {"status": "pong"}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc

    if x_github_event == "pull_request":
        if payload.get("action") not in REVIEW_ACTIONS:
            return {"status": "ignored", "reason": f"action {payload.get('action')}"}
        if payload["pull_request"].get("draft"):
            return {"status": "ignored", "reason": "draft"}
        job = build_job(payload)
        if settings.queue_mode == "redis":
            enqueue(job, settings)
        else:
            background.add_task(process_review_safe, job)
        return {"status": "queued", "pr": job["pr_number"], "sha": job["head_sha"][:7]}

    if x_github_event == "pull_request_review_thread":
        # A thread on one of our comments being resolved is the strongest "this was useful" signal.
        first = (payload.get("thread", {}).get("comments") or [{}])[0]
        comment_id = first.get("id")
        if comment_id and storage.known_comment(int(comment_id)):
            signal = "resolved" if payload.get("action") == "resolved" else "unresolved"
            storage.record_feedback(int(comment_id), signal)
            return {"status": "feedback", "signal": signal}
        return {"status": "ignored"}

    return {"status": "ignored", "reason": f"event {x_github_event}"}
