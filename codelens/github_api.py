"""GitHub App authentication and the handful of REST calls we need."""
from __future__ import annotations

import logging
import time

import httpx
import jwt

from .config import Settings
from .schemas import ChangedFile, ReviewComment

log = logging.getLogger(__name__)
API_VERSION = "2022-11-28"
SEVERITY_EMOJI = {"critical": "🛑", "high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {message[:300]}")
        self.status = status


def render_comment_body(c: ReviewComment) -> str:
    emoji = SEVERITY_EMOJI.get(c.severity, "")
    parts = [f"**{emoji} {c.severity.upper()} · {c.category}** — {c.title}".strip(), "", c.explanation.strip()]
    if c.related_files:
        parts += ["", "Related: " + ", ".join(f"`{p}`" for p in c.related_files[:5])]
    if c.suggestion is not None:
        fence = "````" if "```" in c.suggestion else "```"
        parts += ["", f"{fence}suggestion", c.suggestion.rstrip("\n"), fence]
    parts += ["", f"<sub>CodeLens · confidence {c.confidence:.2f} · react 👍 / 👎 to help tune future reviews</sub>"]
    return "\n".join(parts)


class InstallationClient:
    def __init__(self, token: str, settings: Settings, http: httpx.Client) -> None:
        self.token, self.settings, self.http = token, settings, http

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        resp = self.http.request(method, f"{self.settings.github_api_url}{path}", headers=self.headers, **kw)
        if resp.status_code >= 400:
            raise GitHubError(resp.status_code, resp.text)
        return resp

    def clone_url(self, owner: str, repo: str) -> str:
        return f"{self.settings.github_clone_host}/{owner}/{repo}.git"

    def get_pr(self, owner: str, repo: str, number: int) -> dict:
        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}").json()

    def list_pr_files(self, owner: str, repo: str, number: int) -> list[ChangedFile]:
        files: list[ChangedFile] = []
        page = 1
        while True:
            data = self._request(
                "GET", f"/repos/{owner}/{repo}/pulls/{number}/files", params={"per_page": 100, "page": page}
            ).json()
            for f in data:
                files.append(ChangedFile(
                    path=f["filename"], status=f["status"], patch=f.get("patch"),
                    additions=f.get("additions", 0), deletions=f.get("deletions", 0),
                ))
            if len(data) < 100 or page >= 30:
                return files
            page += 1

    def create_review(self, owner: str, repo: str, number: int, commit_id: str, body: str,
                      comments: list[ReviewComment]) -> dict:
        payload = {
            "commit_id": commit_id, "body": body, "event": "COMMENT",
            "comments": [self._comment_payload(c) for c in comments],
        }
        return self._request("POST", f"/repos/{owner}/{repo}/pulls/{number}/reviews", json=payload).json()

    def create_single_comment(self, owner: str, repo: str, number: int, commit_id: str, c: ReviewComment) -> dict:
        payload = {"commit_id": commit_id, **self._comment_payload(c)}
        return self._request("POST", f"/repos/{owner}/{repo}/pulls/{number}/comments", json=payload).json()

    @staticmethod
    def _comment_payload(c: ReviewComment) -> dict:
        p = {"path": c.path, "line": c.line, "side": "RIGHT", "body": render_comment_body(c)}
        if c.start_line is not None and c.start_line < c.line:
            p["start_line"], p["start_side"] = c.start_line, "RIGHT"
        return p

    def list_review_comments(self, owner: str, repo: str, number: int, review_id: int) -> list[dict]:
        return self._request(
            "GET", f"/repos/{owner}/{repo}/pulls/{number}/reviews/{review_id}/comments", params={"per_page": 100}
        ).json()

    def comment_reactions(self, owner: str, repo: str, comment_id: int) -> list[dict]:
        return self._request(
            "GET", f"/repos/{owner}/{repo}/pulls/comments/{comment_id}/reactions", params={"per_page": 100}
        ).json()


class GitHubApp:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self.http = http or httpx.Client(timeout=30)
        self._tokens: dict[int, tuple[str, float]] = {}

    def app_jwt(self) -> str:
        if not self.settings.github_app_id or not self.settings.github_private_key:
            raise RuntimeError("GITHUB_APP_ID / GITHUB_PRIVATE_KEY are not configured")
        now = int(time.time())
        return jwt.encode(
            {"iat": now - 60, "exp": now + 9 * 60, "iss": self.settings.github_app_id},
            self.settings.github_private_key, algorithm="RS256",
        )

    def installation_token(self, installation_id: int) -> str:
        cached = self._tokens.get(installation_id)
        if cached and cached[1] - time.time() > 120:
            return cached[0]
        resp = self.http.post(
            f"{self.settings.github_api_url}/app/installations/{installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {self.app_jwt()}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": API_VERSION},
        )
        if resp.status_code >= 400:
            raise GitHubError(resp.status_code, resp.text)
        data = resp.json()
        expires = time.mktime(time.strptime(data["expires_at"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        self._tokens[installation_id] = (data["token"], expires)
        return data["token"]

    def client_for(self, installation_id: int) -> InstallationClient:
        return InstallationClient(self.installation_token(installation_id), self.settings, self.http)
