"""The agent as a LangGraph state machine.

    START ──> agent ──(tool calls)──> tools ──> agent ──> ... ──(done)──> END

`agent` calls Claude through LangChain's ChatAnthropic; `tools` runs the requested
tool calls; a SQLite checkpointer stores every step so sessions can be resumed.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ..prompts import TRUNCATED_TOOL_CALL
from ..utils.logger import get_logger
from .executor import ToolExecutor
from .memory import validate_session_id
from .state import AgentResult, AgentState, Usage

log = get_logger("agent")

# on_event(kind, payload): kind is "text" | "tool_call" | "tool_result"
EventFn = Callable[[str, Any], None]


def _noop(_kind: str, _payload: Any) -> None:
    pass


def _stop_reason(msg: AIMessage) -> str | None:
    return (msg.response_metadata or {}).get("stop_reason")


def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(
        b["text"] for b in msg.content if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
    )


class CodingAgent:
    def __init__(
        self,
        llm: Any,
        executor: ToolExecutor,
        system_prompt: str,
        checkpointer: Any,
        max_steps: int = 30,
        on_event: EventFn = _noop,
    ):
        self.llm = llm
        self.executor = executor
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_event = on_event
        self.checkpointer = checkpointer
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph nodes
    def _agent_node(self, state: AgentState) -> dict:
        messages = [SystemMessage(self.system_prompt), *state["messages"]]
        response = self.llm.invoke(messages)
        log.info("stop_reason=%s tool_calls=%d", _stop_reason(response), len(response.tool_calls))
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def _tools_node(self, state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        if _stop_reason(last) == "max_tokens":
            # The response was cut off mid tool call, so don't run anything. Every
            # tool_use still needs a result: report the truncation so the model retries.
            calls = [*last.tool_calls, *last.invalid_tool_calls]
            return {"messages": [
                ToolMessage(content=TRUNCATED_TOOL_CALL, tool_call_id=c["id"], name=c.get("name") or "", status="error")
                for c in calls if c.get("id")
            ]}
        return {"messages": [self.executor.execute(call) for call in last.tool_calls]}

    # ------------------------------------------------------------------ routing
    def _after_agent(self, state: AgentState) -> str:
        last: AIMessage = state["messages"][-1]
        stop = _stop_reason(last)
        if last.tool_calls or (stop == "max_tokens" and last.invalid_tool_calls):
            return "tools"  # always answer tool calls so the history stays valid
        if stop == "pause_turn" and state.get("steps", 0) < self.max_steps:
            return "agent"  # server paused a long turn; call again to resume
        return END

    def _after_tools(self, state: AgentState) -> str:
        return END if state.get("steps", 0) >= self.max_steps else "agent"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._after_agent, ["tools", "agent", END])
        graph.add_conditional_edges("tools", self._after_tools, ["agent", END])
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ public API
    def session_exists(self, session_id: str) -> bool:
        config = {"configurable": {"thread_id": validate_session_id(session_id)}}
        return bool(self.graph.get_state(config).values.get("messages"))

    def messages(self, session_id: str) -> list[Any]:
        config = {"configurable": {"thread_id": validate_session_id(session_id)}}
        return self.graph.get_state(config).values.get("messages", [])

    def run(self, task: str, session_id: str | None = None) -> AgentResult:
        session_id = validate_session_id(session_id) if session_id else uuid.uuid4().hex[:12]
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.max_steps * 2 + 10,
        }
        before = len(self.messages(session_id))
        inputs = {"messages": [HumanMessage(task)], "steps": 0}

        for update in self.graph.stream(inputs, config, stream_mode="updates"):
            for node, delta in update.items():
                self._emit(node, (delta or {}).get("messages", []))

        values = self.graph.get_state(config).values
        new_messages = values["messages"][before:]
        return AgentResult(
            status=self._status(values["messages"][-1]),
            final_text=self._final_text(new_messages),
            steps=values.get("steps", 0),
            session_id=session_id,
            usage=Usage.from_messages(new_messages),
        )

    # ------------------------------------------------------------------ helpers
    def _emit(self, node: str, messages: list[Any]) -> None:
        for msg in messages:
            if isinstance(msg, AIMessage):
                if text := _text(msg):
                    self.on_event("text", text)
                for call in msg.tool_calls:
                    self.on_event("tool_call", call)
            elif isinstance(msg, ToolMessage):
                self.on_event("tool_result", msg)

    @staticmethod
    def _status(last: Any) -> str:
        if isinstance(last, ToolMessage):
            return "max_steps"
        stop = _stop_reason(last) if isinstance(last, AIMessage) else None
        return {
            "end_turn": "completed",
            "refusal": "refusal",
            "max_tokens": "max_tokens",
            "pause_turn": "max_steps",
        }.get(stop, "error")

    @staticmethod
    def _final_text(messages: list[Any]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and (text := _text(msg)):
                return text
        return ""
