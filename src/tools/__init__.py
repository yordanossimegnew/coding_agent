"""Tool registry for the coding agent."""

from __future__ import annotations

from . import filesystem, search, shell
from .base import Tool, ToolError, Workspace


def default_tools() -> list[Tool]:
    return [*filesystem.TOOLS, *search.TOOLS, *shell.TOOLS]


__all__ = ["Tool", "ToolError", "Workspace", "default_tools"]
