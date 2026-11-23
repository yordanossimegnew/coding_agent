"""Graph state, token usage and run results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph graph (checkpointed per session)."""

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int  # model calls in the current run


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_messages(cls, messages: list[Any]) -> "Usage":
        usage = cls()
        for msg in messages:
            meta = getattr(msg, "usage_metadata", None) if isinstance(msg, AIMessage) else None
            if not meta:
                continue
            details = meta.get("input_token_details") or {}
            usage.input_tokens += meta.get("input_tokens", 0)
            usage.output_tokens += meta.get("output_tokens", 0)
            usage.cache_read_input_tokens += details.get("cache_read", 0) or 0
            usage.cache_creation_input_tokens += details.get("cache_creation", 0) or 0
        return usage


@dataclass
class AgentResult:
    status: str  # completed | max_steps | max_tokens | refusal | error
    final_text: str
    steps: int
    session_id: str
    usage: Usage = field(default_factory=Usage)
