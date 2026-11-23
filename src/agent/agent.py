"""The agent loop: ask the model, run the tools it calls, repeat until it is done."""

from __future__ import annotations

from typing import Any, Callable

from ..prompts import TRUNCATED_TOOL_CALL
from ..utils.logger import get_logger
from .executor import ToolExecutor
from .memory import SessionMemory
from .state import AgentResult, AgentState

log = get_logger("agent")

# on_event(kind, payload): kind is "text" | "tool_call" | "tool_result"
EventFn = Callable[[str, Any], None]


def _noop(_kind: str, _payload: Any) -> None:
    pass


def _blocks(response: Any, block_type: str) -> list[Any]:
    return [b for b in response.content if getattr(b, "type", None) == block_type]


class CodingAgent:
    def __init__(
        self,
        llm: Any,
        executor: ToolExecutor,
        system_prompt: str,
        max_steps: int = 30,
        memory: SessionMemory | None = None,
        on_event: EventFn = _noop,
    ):
        self.llm = llm
        self.executor = executor
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.memory = memory
        self.on_event = on_event

    def run(self, task: str, state: AgentState | None = None) -> AgentResult:
        state = state or AgentState()
        state.add_user_text(task)
        try:
            return self._loop(state)
        finally:
            if self.memory is not None:
                self.memory.save(state)

    def _loop(self, state: AgentState) -> AgentResult:
        tools = self.executor.definitions()
        last_text = ""
        for _ in range(self.max_steps):
            state.steps += 1
            response = self.llm.create(system=self.system_prompt, messages=state.messages, tools=tools)
            state.usage.add(response.usage)
            state.add_assistant(response.content)

            texts = [b.text for b in _blocks(response, "text")]
            if texts:
                last_text = "\n".join(texts)
                self.on_event("text", last_text)

            stop = response.stop_reason
            tool_uses = _blocks(response, "tool_use")

            if stop == "tool_use":
                results = []
                for tool_use in tool_uses:
                    self.on_event("tool_call", tool_use)
                    result = self.executor.execute(tool_use)
                    self.on_event("tool_result", result)
                    results.append(result)
                state.add_tool_results(results)
                continue

            if stop == "max_tokens" and tool_uses:
                # The last tool call was cut off mid-generation. Every tool_use needs a
                # tool_result, so report the truncation and let the model retry smaller.
                state.add_tool_results([
                    {"type": "tool_result", "tool_use_id": t.id, "content": TRUNCATED_TOOL_CALL, "is_error": True}
                    for t in tool_uses
                ])
                continue

            if stop == "pause_turn":  # server paused a long turn; re-send to resume
                continue

            if stop == "end_turn":
                return self._result("completed", last_text, state)
            if stop == "refusal":
                details = getattr(response, "stop_details", None)
                log.warning("Model declined: %s", details)
                return self._result("refusal", last_text or "The model declined this request.", state)
            if stop == "max_tokens":
                return self._result("max_tokens", last_text, state)

            log.warning("Unexpected stop_reason: %s", stop)
            return self._result("error", last_text or f"Stopped: {stop}", state)

        return self._result("max_steps", last_text or "Stopped: step limit reached.", state)

    @staticmethod
    def _result(status: str, text: str, state: AgentState) -> AgentResult:
        return AgentResult(
            status=status, final_text=text, steps=state.steps, session_id=state.session_id, usage=state.usage
        )
