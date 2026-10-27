"""Runtime configuration loaded from environment variables or a .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_META_API_URL = "https://api.meta.ai/v1"
DEFAULT_META_MODEL = "muse-spark-1.3"


def load_project_env() -> None:
    """Load `.env` into the process environment without overriding variables already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def meta_env() -> dict[str, str]:
    """Read the Meta Model API settings directly from the environment."""
    load_project_env()
    return {
        "url": os.getenv("META_API_URL", DEFAULT_META_API_URL).strip() or DEFAULT_META_API_URL,
        "key": os.getenv("META_API_KEY", "").strip(),
        "model": os.getenv("META_MODEL", DEFAULT_META_MODEL).strip() or DEFAULT_META_MODEL,
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    upstream_base_url: str = "http://localhost:9000"
    redis_url: str = "redis://localhost:6379/0"
    fail_mode: Literal["open", "closed"] = "open"

    meta_api_key: str = ""
    meta_api_url: str = DEFAULT_META_API_URL
    meta_model: str = DEFAULT_META_MODEL
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = Field(4096, gt=0)
    llm_max_retries: int = Field(2, ge=0)
    # Skip re-analysis of the same client and route template within this window
    agent_cooldown_seconds: float = Field(30.0, ge=0.0)
    agent_queue_size: int = Field(1000, gt=0)

    suspicious_threshold: float = Field(0.45, ge=0.0, le=1.0)
    hard_block_threshold: float = Field(0.9, ge=0.0, le=1.0)

    rate_window_seconds: int = Field(10, gt=0)
    rate_limit_per_window: int = Field(50, gt=0)
    # Distinct object IDs one client may touch on a single route template before BOLA suspicion
    bola_window_seconds: int = Field(60, gt=0)
    bola_distinct_threshold: int = Field(15, gt=0)
    history_size: int = Field(25, gt=0)
    trust_forwarded_for: bool = False

    block_confidence: float = Field(0.85, ge=0.0, le=1.0)
    critical_threat_confidence: float = Field(0.7, ge=0.0, le=1.0)
    block_ttl_seconds: int = Field(3600, gt=0)
    # When false, blocklist decisions are logged but not written or enforced. For local testing
    # from a single machine. Production must leave this true.
    enforce_blocks: bool = True

    admin_token: str = "change-me"
    max_body_bytes: int = Field(1_048_576, gt=0)
    max_events: int = Field(500, gt=0)
    max_agent_concurrency: int = Field(4, gt=0)

    chroma_dir: str = "data/chroma"
    enable_chroma: bool = True

    @model_validator(mode="before")
    @classmethod
    def _meta_from_environ(cls, data: Any) -> Any:
        """Prefer META_API_URL, META_API_KEY, and META_MODEL read with os.getenv."""
        values = dict(data) if isinstance(data, dict) else {}
        env = meta_env()
        values.setdefault("meta_api_url", env["url"])
        values.setdefault("meta_api_key", env["key"])
        values.setdefault("meta_model", env["model"])
        return values

    @property
    def llm_enabled(self) -> bool:
        return bool(self.meta_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
