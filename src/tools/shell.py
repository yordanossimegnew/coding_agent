"""Shell tool: run a command inside the workspace. Gated behind human approval."""

from __future__ import annotations

import subprocess

from ..utils.helpers import truncate
from .base import Tool, ToolError, Workspace, schema


def run_command(workspace: Workspace, command: str) -> str:
    timeout = workspace.command_timeout
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"Command timed out after {timeout}s.") from exc
    output = (
        f"exit_code: {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout or '(empty)'}\n"
        f"--- stderr ---\n{proc.stderr or '(empty)'}"
    )
    return truncate(output)


TOOLS = [
    Tool(
        name="run_command",
        description=(
            "Run a shell command with the workspace as the working directory (e.g. run tests or a "
            "script). Returns exit code, stdout and stderr. The user may decline the command."
        ),
        input_schema=schema({"command": {"type": "string", "description": "The shell command to run."}}),
        handler=run_command,
        requires_approval=True,
    ),
]
