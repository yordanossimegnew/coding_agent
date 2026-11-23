"""Code search: regex grep across the workspace (pure Python, no ripgrep needed)."""

from __future__ import annotations

import re

from .base import Tool, ToolError, Workspace, schema
from .filesystem import IGNORED_DIRS

MAX_MATCHES = 200


def search_code(workspace: Workspace, pattern: str, path: str) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"Invalid regex: {exc}") from exc
    root = workspace.resolve(path)
    if not root.exists():
        raise ToolError(f"Path not found: {path}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())

    matches: list[str] = []
    for file in files:
        if any(part in IGNORED_DIRS for part in file.relative_to(workspace.root).parts):
            continue
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # skip binary / unreadable files
        for lineno, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{workspace.display(file)}:{lineno}: {line.strip()}")
                if len(matches) >= MAX_MATCHES:
                    return "\n".join(matches) + f"\n... (stopped after {MAX_MATCHES} matches)"
    return "\n".join(matches) if matches else "No matches."


TOOLS = [
    Tool(
        name="search_code",
        description="Search file contents with a Python regular expression. Returns path:line: text matches.",
        input_schema=schema({
            "pattern": {"type": "string", "description": "Python regex, e.g. 'def \\w+_test'."},
            "path": {"type": "string", "description": "File or directory to search, relative to the workspace ('.' for all)."},
        }),
        handler=search_code,
    ),
]
