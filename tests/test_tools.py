from __future__ import annotations

import sys

import pytest

from src.tools import ToolError, default_tools
from src.tools.filesystem import edit_file, list_files, read_file, write_file
from src.tools.search import search_code
from src.tools.shell import run_command


def test_write_then_read_with_line_numbers(workspace):
    assert "Created" in write_file(workspace, "pkg/a.py", "x = 1\ny = 2\n")
    out = read_file(workspace, "pkg/a.py")
    assert "    1  x = 1" in out and "    2  y = 2" in out


def test_path_escape_is_blocked(workspace):
    with pytest.raises(ToolError, match="escapes the workspace"):
        read_file(workspace, "../secret.txt")


def test_edit_requires_unique_match(workspace):
    write_file(workspace, "a.py", "foo\nfoo\n")
    with pytest.raises(ToolError, match="matches 2 places"):
        edit_file(workspace, "a.py", "foo", "bar")
    with pytest.raises(ToolError, match="not found"):
        edit_file(workspace, "a.py", "missing", "bar")
    write_file(workspace, "b.py", "alpha\nbeta\n")
    edit_file(workspace, "b.py", "beta", "gamma")
    assert (workspace.root / "b.py").read_text() == "alpha\ngamma\n"


def test_list_files_skips_ignored_dirs(workspace):
    write_file(workspace, "src/main.py", "")
    write_file(workspace, "__pycache__/junk.pyc", "")
    listing = list_files(workspace, ".")
    assert "src/main.py" in listing
    assert "__pycache__" not in listing


def test_search_code(workspace):
    write_file(workspace, "a.py", "def hello():\n    return 1\n")
    assert "a.py:1: def hello():" in search_code(workspace, r"def \w+", ".")
    assert search_code(workspace, "nomatch", ".") == "No matches."
    with pytest.raises(ToolError, match="Invalid regex"):
        search_code(workspace, "(", ".")


def test_run_command_reports_exit_code(workspace):
    out = run_command(workspace, f'"{sys.executable}" -c "print(42)"')
    assert "exit_code: 0" in out and "42" in out


def test_tool_schemas_are_strict():
    for tool in default_tools():
        api = tool.to_api()
        assert api["strict"] is True
        assert api["input_schema"]["additionalProperties"] is False
        assert set(api["input_schema"]["required"]) == set(api["input_schema"]["properties"])
