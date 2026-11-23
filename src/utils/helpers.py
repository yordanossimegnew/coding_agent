"""Small shared helpers."""

from __future__ import annotations

MAX_TOOL_OUTPUT_CHARS = 20_000


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Keep the head and tail of long output so the model sees both ends."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"
