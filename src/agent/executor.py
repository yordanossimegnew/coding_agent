"""Runs tool calls requested by the model and turns them into tool_result blocks."""

from __future__ import annotations

from typing import Any, Callable

from ..tools.base import Tool, ToolError, Workspace
from ..utils.helpers import truncate
from ..utils.logger import get_logger

log = get_logger("executor")

# approve(tool_name, tool_input) -> True to run, False to decline
ApprovalFn = Callable[[str, dict], bool]


def deny_all(_name: str, _input: dict) -> bool:
    return False


def allow_all(_name: str, _input: dict) -> bool:
    return True


class ToolExecutor:
    def __init__(self, tools: list[Tool], workspace: Workspace, approve: ApprovalFn = deny_all):
        self.tools = {tool.name: tool for tool in tools}
        self.workspace = workspace
        self.approve = approve

    def definitions(self) -> list[dict]:
        return [tool.to_api() for tool in self.tools.values()]

    def execute(self, tool_use: Any) -> dict:
        """Run one tool_use block. Never raises: failures become is_error results."""
        name, tool_input = tool_use.name, tool_use.input
        result = {"type": "tool_result", "tool_use_id": tool_use.id}
        tool = self.tools.get(name)
        if tool is None:
            return {**result, "content": f"Unknown tool: {name}", "is_error": True}
        if not isinstance(tool_input, dict):
            return {**result, "content": "Tool input must be a JSON object.", "is_error": True}
        if tool.requires_approval and not self.approve(name, tool_input):
            log.info("User declined %s %s", name, tool_input)
            return {**result, "content": "The user declined to run this.", "is_error": True}
        try:
            output = tool.handler(self.workspace, **tool_input)
            log.info("%s ok", name)
            return {**result, "content": truncate(output)}
        except ToolError as exc:
            log.info("%s tool error: %s", name, exc)
            return {**result, "content": str(exc), "is_error": True}
        except TypeError as exc:  # wrong / missing arguments
            return {**result, "content": f"Invalid arguments: {exc}", "is_error": True}
        except Exception as exc:  # unexpected failure: report it, keep the agent alive
            log.exception("%s crashed", name)
            return {**result, "content": f"{type(exc).__name__}: {exc}", "is_error": True}

    def execute_all(self, tool_uses: list[Any]) -> list[dict]:
        return [self.execute(tool_use) for tool_use in tool_uses]
