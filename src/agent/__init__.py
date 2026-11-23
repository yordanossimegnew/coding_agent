"""Core agent package plus a factory that wires everything together."""

from __future__ import annotations

from typing import Any

from ..models import LLMClient
from ..prompts import build_system_prompt
from ..tools import Workspace, default_tools
from ..utils.config import Settings
from .agent import CodingAgent, EventFn, _noop
from .executor import ApprovalFn, ToolExecutor, allow_all, deny_all
from .memory import SessionMemory
from .state import AgentResult, AgentState


def build_agent(
    settings: Settings,
    approve: ApprovalFn | None = None,
    on_event: EventFn = _noop,
    llm: Any | None = None,
) -> CodingAgent:
    if approve is None:
        approve = allow_all if settings.auto_approve else deny_all
    workspace = Workspace(settings.workspace, command_timeout=settings.command_timeout)
    executor = ToolExecutor(default_tools(), workspace, approve=approve)
    return CodingAgent(
        llm=llm or LLMClient(settings),
        executor=executor,
        system_prompt=build_system_prompt(settings.knowledge_base_dir),
        max_steps=settings.max_steps,
        memory=SessionMemory(settings.sessions_dir),
        on_event=on_event,
    )


__all__ = [
    "AgentResult", "AgentState", "CodingAgent", "SessionMemory", "ToolExecutor",
    "allow_all", "build_agent", "deny_all",
]
