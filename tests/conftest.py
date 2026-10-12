import pytest

from codelens import config


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "testsecret")
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "hash")
    monkeypatch.setenv("QUEUE_MODE", "inline")
    monkeypatch.delenv("GITHUB_PRIVATE_KEY_PATH", raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
