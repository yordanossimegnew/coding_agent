"""File tools: list, read, write, and exact-match edit."""

from __future__ import annotations

from .base import Tool, ToolError, Workspace, schema

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
MAX_LIST_ENTRIES = 500


def list_files(workspace: Workspace, path: str) -> str:
    target = workspace.resolve(path)
    if not target.exists():
        raise ToolError(f"Directory not found: {path}")
    if not target.is_dir():
        raise ToolError(f"Not a directory: {path}")
    entries: list[str] = []
    for item in sorted(target.rglob("*")):
        if any(part in IGNORED_DIRS for part in item.relative_to(target).parts):
            continue
        suffix = "/" if item.is_dir() else ""
        entries.append(workspace.display(item) + suffix)
        if len(entries) >= MAX_LIST_ENTRIES:
            entries.append(f"... (stopped after {MAX_LIST_ENTRIES} entries)")
            break
    return "\n".join(entries) if entries else "(empty directory)"


def read_file(workspace: Workspace, path: str) -> str:
    target = workspace.resolve(path)
    if not target.is_file():
        raise ToolError(f"File not found: {path}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"{path} is not a UTF-8 text file.") from exc
    numbered = [f"{i:>5}  {line}" for i, line in enumerate(text.splitlines(), start=1)]
    return "\n".join(numbered) if numbered else "(empty file)"


def write_file(workspace: Workspace, path: str, content: str) -> str:
    target = workspace.resolve(path)
    if target.is_dir():
        raise ToolError(f"{path} is a directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {path} ({len(content.splitlines())} lines)."


def edit_file(workspace: Workspace, path: str, old_text: str, new_text: str) -> str:
    target = workspace.resolve(path)
    if not target.is_file():
        raise ToolError(f"File not found: {path}")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        raise ToolError("old_text was not found. Re-read the file and copy the text exactly.")
    if count > 1:
        raise ToolError(f"old_text matches {count} places. Include more surrounding lines.")
    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}."


TOOLS = [
    Tool(
        name="list_files",
        description="Recursively list files and folders under a workspace directory. Use '.' for the workspace root.",
        input_schema=schema({"path": {"type": "string", "description": "Directory relative to the workspace."}}),
        handler=list_files,
    ),
    Tool(
        name="read_file",
        description="Read a UTF-8 text file. Output is prefixed with line numbers (not part of the file).",
        input_schema=schema({"path": {"type": "string", "description": "File path relative to the workspace."}}),
        handler=read_file,
    ),
    Tool(
        name="write_file",
        description="Create a file or completely overwrite it. Prefer edit_file for changes to existing files.",
        input_schema=schema({
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "content": {"type": "string", "description": "The full new file content."},
        }),
        handler=write_file,
    ),
    Tool(
        name="edit_file",
        description=(
            "Replace one exact occurrence of old_text with new_text in a file. old_text must match "
            "the file exactly (whitespace included, no line-number prefixes) and be unique."
        ),
        input_schema=schema({
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
        }),
        handler=edit_file,
    ),
]
