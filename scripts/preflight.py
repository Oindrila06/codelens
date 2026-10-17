"""Check your setup before the first live run. Prints no secrets.

    python scripts/preflight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, installed or not

import shutil

import httpx

from codelens.config import get_settings
from codelens.github_api import GitHubApp
from codelens.llm import AnthropicLLM


def check(label: str, ok: bool, hint: str = "") -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {label}" + (f"\n       -> {hint}" if not ok and hint else ""))
    return ok


def main() -> int:
    s = get_settings()
    ok = True
    print("--- Local tools")
    ok &= check("git installed", bool(shutil.which("git")), "install Git for Windows")
    ok &= check("ruff installed (venv active?)", bool(shutil.which("ruff")),
                "run .venv\\Scripts\\Activate.ps1 first, then pip install -r requirements.txt")
    check("npx available (needed for smee-client)", bool(shutil.which("npx")), "install Node.js LTS from nodejs.org")

    print("--- .env values")
    ok &= check("GITHUB_APP_ID is set", bool(s.github_app_id), "add GITHUB_APP_ID to .env")
    key_ok = s.github_private_key.strip().startswith("-----BEGIN")
    ok &= check("Private key file was read", key_ok,
                "GITHUB_PRIVATE_KEY_PATH must point to the .pem (run from the project folder)")
    ok &= check("Webhook secret is set", len(s.github_webhook_secret) >= 16,
                "GITHUB_WEBHOOK_SECRET must equal the secret in the GitHub App settings")
    ok &= check("ANTHROPIC_API_KEY is set", bool(s.anthropic_api_key), "add your key to .env")

    print("--- GitHub")
    if s.github_app_id and key_ok:
        try:
            jwt_token = GitHubApp(s).app_jwt()
            headers = {"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
            me = httpx.get(f"{s.github_api_url}/app", headers=headers, timeout=20)
            if me.status_code == 200:
                check(f"Authenticated as GitHub App '{me.json().get('slug')}'", True)
                inst = httpx.get(f"{s.github_api_url}/app/installations", headers=headers, timeout=20).json()
                names = [i["account"]["login"] for i in inst] if isinstance(inst, list) else []
                ok &= check(f"App is installed on: {', '.join(names) or 'nothing'}", bool(names),
                            "App settings -> Install App -> install it on your test repo")
            else:
                ok &= check("GitHub accepted the App credentials", False,
                            f"HTTP {me.status_code}: wrong App ID, or the key belongs to a different App")
        except Exception as exc:  # noqa: BLE001
            ok &= check("GitHub App authentication", False, f"{type(exc).__name__}: {str(exc)[:150]}")
    else:
        print("(skipped: App ID or private key missing)")

    print("--- LLM")
    if s.anthropic_api_key:
        try:
            out = AnthropicLLM(s).complete_json("Reply with JSON only.", 'Return {"ok": true}', max_tokens=30)
            ok &= check(f"Model '{s.llm_model}' answered", bool(out), "")
        except Exception as exc:  # noqa: BLE001
            ok &= check(f"Model '{s.llm_model}' answered", False,
                        f"{type(exc).__name__}: {str(exc)[:200]}  (try another LLM_MODEL in .env)")
    else:
        print("(skipped: no API key)")

    print("\nAll good. Start the tunnel and the server." if ok else "\nFix the FAIL items above, then run this again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
