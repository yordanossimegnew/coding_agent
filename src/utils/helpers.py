"""Small shared helpers."""

from __future__ import annotations

from typing import Any

MAX_TOOL_OUTPUT_CHARS = 20_000


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Keep the head and tail of long output so the model sees both ends."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"


def to_jsonable(block: Any) -> Any:
    """Convert an SDK content block (pydantic model) into a plain dict for storage."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    if hasattr(block, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(block).items()}
    return block


def serialize_messages(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = [to_jsonable(b) for b in content]
        out.append({"role": message["role"], "content": content})
    return out
