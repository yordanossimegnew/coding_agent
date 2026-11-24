"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5-5"
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_steps: int = 30
    workspace: Path = PROJECT_ROOT / "workspace"
    command_timeout: int = 60
    auto_approve: bool = False
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/coding_agent"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"


def load_settings(**overrides) -> Settings:
    """Build Settings from env vars, then apply any non-None keyword overrides."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        model=os.getenv("AGENT_MODEL", Settings.model),
        effort=os.getenv("AGENT_EFFORT", Settings.effort),
        max_tokens=int(os.getenv("AGENT_MAX_TOKENS", Settings.max_tokens)),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", Settings.max_steps)),
        workspace=Path(os.getenv("AGENT_WORKSPACE", str(Settings.workspace))),
        command_timeout=int(os.getenv("AGENT_COMMAND_TIMEOUT", Settings.command_timeout)),
        auto_approve=_env_bool("AGENT_AUTO_APPROVE", Settings.auto_approve),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
    )
    clean = {k: v for k, v in overrides.items() if v is not None}
    if "workspace" in clean:
        clean["workspace"] = Path(clean["workspace"])
    return replace(settings, **clean)
