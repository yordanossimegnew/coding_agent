"""Runs tool calls requested by the model and turns them into ToolMessages."""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import ToolMessage

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

    def execute(self, call: dict) -> ToolMessage:
        """Run one tool call ({"name", "args", "id"}). Never raises: failures become error results."""
        name, args, call_id = call["name"], call.get("args"), call["id"]

        def error(text: str) -> ToolMessage:
            return ToolMessage(content=text, tool_call_id=call_id, name=name, status="error")

        tool = self.tools.get(name)
        if tool is None:
            return error(f"Unknown tool: {name}")
        if not isinstance(args, dict):
            return error("Tool input must be a JSON object.")
        if tool.requires_approval and not self.approve(name, args):
            log.info("User declined %s %s", name, args)
            return error("The user declined to run this.")
        try:
            output = tool.handler(self.workspace, **args)
            log.info("%s ok", name)
            return ToolMessage(content=truncate(output), tool_call_id=call_id, name=name)
        except ToolError as exc:
            log.info("%s tool error: %s", name, exc)
            return error(str(exc))
        except TypeError as exc:  # wrong / missing arguments
            return error(f"Invalid arguments: {exc}")
        except Exception as exc:  # unexpected failure: report it, keep the agent alive
            log.exception("%s crashed", name)
            return error(f"{type(exc).__name__}: {exc}")
