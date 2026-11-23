"""Tool primitives: the Tool definition, ToolError, and the sandboxed Workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """Raised by a tool to report a recoverable error back to the model (is_error=True)."""


class Workspace:
    """A directory the agent is confined to. Every path is resolved inside it."""

    def __init__(self, root: Path, command_timeout: int = 60):
        self.root = Path(root).resolve()
        self.command_timeout = command_timeout
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise ToolError(f"Path '{relative}' escapes the workspace; use paths inside it.")
        return candidate

    def display(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return rel or "."


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]  # handler(workspace, **input) -> str
    requires_approval: bool = False

    def to_api(self) -> dict[str, Any]:
        """The tool definition sent to the Messages API (strict schema validation on)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


def schema(properties: dict[str, dict]) -> dict[str, Any]:
    """Build a strict JSON schema where every listed property is required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
