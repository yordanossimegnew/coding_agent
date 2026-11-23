from __future__ import annotations

import pytest

from src.tools import Workspace
from src.utils.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        workspace=tmp_path / "ws",
        sessions_dir=tmp_path / "sessions",
        knowledge_base_dir=tmp_path / "kb",
        log_dir=tmp_path / "logs",
        max_steps=5,
    )


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "ws", command_timeout=10)
