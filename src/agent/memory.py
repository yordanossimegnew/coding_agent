"""Session memory: a LangGraph PostgreSQL checkpointer, one thread per session."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> str:
    if not _SAFE_ID.match(session_id):
        raise ValueError(f"Invalid session id: {session_id!r}")
    return session_id


def open_checkpointer(database_url: str) -> PostgresSaver:
    """Connection-pooled LangGraph checkpointer; creates its tables on first use."""
    conninfo = database_url.replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(conninfo, min_size=1, max_size=10, open=True,
                          kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def list_sessions(checkpointer: PostgresSaver) -> list[str]:
    """Thread ids, most recently updated first."""
    seen: list[str] = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.append(thread_id)
    return seen


def _text_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [
        {"type": "text", "text": block["text"]}
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]


def to_transcript(messages: list[Any]) -> list[dict]:
    """Convert LangChain messages into simple role/content dicts for the API and UI.

    Shape mirrors the Claude API: assistant turns hold text + tool_use blocks, and the
    tool results that follow are grouped into one user turn of tool_result blocks.
    """
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            blocks = _text_blocks(msg.content)
            blocks += [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                for c in msg.tool_calls
            ]
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(msg, ToolMessage):
            block = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": str(msg.content)}
            if msg.status == "error":
                block["is_error"] = True
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out
