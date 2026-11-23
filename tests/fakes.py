"""A scripted stand-in for the chat model so tests run offline, with no API key."""

from __future__ import annotations

from langchain_core.messages import AIMessage


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, **args) -> dict:
    return {"type": "tool_call", "id": id_, "name": name, "args": args}


def response(*parts, stop_reason: str = "end_turn") -> AIMessage:
    """Build an AIMessage like ChatAnthropic returns: text content + tool_calls."""
    return AIMessage(
        content=[p for p in parts if p["type"] == "text"],
        tool_calls=[p for p in parts if p["type"] == "tool_call"],
        response_metadata={"stop_reason": stop_reason},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)
