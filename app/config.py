"""Runtime configuration loaded from environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    upstream_base_url: str = "http://localhost:9000"
    redis_url: str = "redis://localhost:6379/0"
    fail_mode: Literal["open", "closed"] = "open"

    baseten_api_key: str = ""
    baseten_base_url: str = "https://inference.baseten.co/v1"
    baseten_model: str = "meta-models/muse-spark-open-1026"
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

    admin_token: str = "change-me"
    max_body_bytes: int = Field(1_048_576, gt=0)
    max_events: int = Field(500, gt=0)
    max_agent_concurrency: int = Field(4, gt=0)

    chroma_dir: str = "data/chroma"
    enable_chroma: bool = True

    @property
    def llm_enabled(self) -> bool:
        return bool(self.baseten_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
