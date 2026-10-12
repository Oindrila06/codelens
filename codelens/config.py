"""Runtime settings, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _load_private_key() -> str:
    path = os.environ.get("GITHUB_PRIVATE_KEY_PATH", "").strip()
    if path and Path(path).exists():
        return Path(path).read_text()
    return os.environ.get("GITHUB_PRIVATE_KEY", "").replace("\\n", "\n")


@dataclass(frozen=True)
class Settings:
    github_app_id: str
    github_webhook_secret: str
    github_private_key: str
    github_api_url: str
    github_clone_host: str
    anthropic_api_key: str
    llm_model: str
    llm_max_tokens: int
    price_in_per_mtok: float
    price_out_per_mtok: float
    data_dir: Path
    queue_mode: str
    redis_url: str
    embeddings_backend: str
    embedding_model: str
    semgrep_config: str
    context_token_budget: int
    max_pr_files: int
    max_file_bytes: int
    triage_model_dir: str
    triage_threshold: float

    @property
    def db_path(self) -> Path:
        return self.data_dir / "codelens.db"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR", "./data")).resolve()
    return Settings(
        github_app_id=os.environ.get("GITHUB_APP_ID", ""),
        github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET", ""),
        github_private_key=_load_private_key(),
        github_api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
        github_clone_host=os.environ.get("GITHUB_CLONE_HOST", "https://github.com").rstrip("/"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
        price_in_per_mtok=_env_float("LLM_PRICE_IN_PER_MTOK", 0.0),
        price_out_per_mtok=_env_float("LLM_PRICE_OUT_PER_MTOK", 0.0),
        data_dir=data_dir,
        queue_mode=os.environ.get("QUEUE_MODE", "inline").lower(),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        embeddings_backend=os.environ.get("EMBEDDINGS_BACKEND", "hash").lower(),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        semgrep_config=os.environ.get("SEMGREP_CONFIG", "p/python"),
        context_token_budget=_env_int("CONTEXT_TOKEN_BUDGET", 6000),
        max_pr_files=_env_int("MAX_PR_FILES", 25),
        max_file_bytes=_env_int("MAX_FILE_BYTES", 300_000),
        triage_model_dir=os.environ.get("TRIAGE_MODEL_DIR", ""),
        triage_threshold=_env_float("TRIAGE_THRESHOLD", 0.4),
    )
