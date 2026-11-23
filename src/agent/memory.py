"""Session memory: persist a conversation to JSON so it can be resumed later."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..utils.helpers import serialize_messages
from .state import AgentState, Usage

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionMemory:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if not _SAFE_ID.match(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        return self.directory / f"{session_id}.json"

    def save(self, state: AgentState) -> Path:
        path = self._path(state.session_id)
        payload = {
            "session_id": state.session_id,
            "steps": state.steps,
            "usage": vars(state.usage),
            "messages": serialize_messages(state.messages),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, session_id: str) -> AgentState | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentState(
            session_id=data["session_id"],
            messages=data["messages"],
            usage=Usage(**data.get("usage", {})),
            steps=data.get("steps", 0),
        )

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))
