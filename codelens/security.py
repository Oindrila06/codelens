"""Webhook signature verification."""
from __future__ import annotations

import hashlib
import hmac


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Validate GitHub's X-Hub-Signature-256 header using a constant-time comparison."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
