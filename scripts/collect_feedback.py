"""Poll GitHub for 👍 / 👎 reactions on comments CodeLens posted and store them as feedback.

Reactions do not trigger webhooks, so run this periodically (cron, or a GitHub Actions schedule):
    python scripts/collect_feedback.py
"""
from __future__ import annotations

import logging

from codelens.config import get_settings
from codelens.github_api import GitHubApp, GitHubError
from codelens.storage import Storage

log = logging.getLogger("collect_feedback")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    storage, app = Storage(settings.db_path), GitHubApp(settings)
    new = 0
    for row in storage.comments_for_polling():
        owner, repo = row["repo"].split("/", 1)
        try:
            client = app.client_for(int(row["installation_id"]))
            reactions = client.comment_reactions(owner, repo, int(row["gh_comment_id"]))
        except GitHubError as exc:
            log.warning("skip comment %s: %s", row["gh_comment_id"], exc)
            continue
        for r in reactions:
            signal = {"+1": "thumbs_up", "-1": "thumbs_down"}.get(r.get("content"))
            if signal and not (r.get("user") or {}).get("login", "").endswith("[bot]"):
                new += storage.record_feedback(int(row["gh_comment_id"]), signal)
    log.info("recorded %d new feedback signal(s)", new)


if __name__ == "__main__":
    main()
