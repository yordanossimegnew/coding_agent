"""Conversation state and run results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        for name in vars(self):
            setattr(self, name, getattr(self, name) + (getattr(usage, name, 0) or 0))


@dataclass
class AgentState:
    """Everything needed to continue a conversation with the model."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content: list[Any]) -> None:
        # Append the full content (thinking + text + tool_use blocks) unchanged;
        # the API expects its own blocks back exactly as it sent them.
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict]) -> None:
        # All results for one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": results})


@dataclass
class AgentResult:
    status: str  # completed | max_steps | max_tokens | refusal | error
    final_text: str
    steps: int
    session_id: str
    usage: Usage
