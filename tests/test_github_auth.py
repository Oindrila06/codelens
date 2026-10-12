import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from codelens.config import get_settings
from codelens.github_api import GitHubApp, GitHubError, InstallationClient


@pytest.fixture
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    return pem, key.public_key()


def settings_with(monkeypatch, pem):
    monkeypatch.setenv("GITHUB_APP_ID", "4242")
    monkeypatch.setenv("GITHUB_PRIVATE_KEY", pem.replace("\n", "\\n"))
    get_settings.cache_clear()
    return get_settings()


def test_app_jwt_is_valid_rs256_with_short_expiry(monkeypatch, keypair):
    pem, public = keypair
    app = GitHubApp(settings_with(monkeypatch, pem))
    claims = jwt.decode(app.app_jwt(), public, algorithms=["RS256"])
    assert claims["iss"] == "4242" and claims["exp"] - claims["iat"] <= 10 * 60 + 60


def test_installation_token_is_cached(monkeypatch, keypair):
    pem, _ = keypair
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        assert request.url.path == "/app/installations/9/access_tokens"
        assert request.headers["Authorization"].startswith("Bearer ey")
        return httpx.Response(201, json={"token": "ghs_abc", "expires_at": "2099-01-01T00:00:00Z"})

    app = GitHubApp(settings_with(monkeypatch, pem), httpx.Client(transport=httpx.MockTransport(handler)))
    assert app.installation_token(9) == "ghs_abc" and app.installation_token(9) == "ghs_abc"
    assert len(calls) == 1


def test_missing_credentials_raise_clear_error(monkeypatch):
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_PRIVATE_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="not configured"):
        GitHubApp(get_settings()).app_jwt()


def test_list_pr_files_paginates_and_errors_surface():
    pages = {1: [{"filename": f"f{i}.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"} for i in range(100)],
             2: [{"filename": "last.py", "status": "added"}]}

    def handler(request: httpx.Request):
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json=pages[int(request.url.params["page"])])
        return httpx.Response(404, json={"message": "nope"})

    client = InstallationClient("t", get_settings(), httpx.Client(transport=httpx.MockTransport(handler)))
    files = client.list_pr_files("o", "r", 1)
    assert len(files) == 101 and files[-1].patch is None
    with pytest.raises(GitHubError) as exc:
        client.get_pr("o", "r", 1)
    assert exc.value.status == 404
