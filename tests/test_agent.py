from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent import build_agent, list_sessions, open_checkpointer, to_transcript
from tests.fakes import FakeLLM, response, text, tool_use


def test_graph_runs_tools_until_done(settings):
    llm = FakeLLM([
        response(
            text("Creating the file."),
            tool_use("t1", "write_file", path="hello.py", content="print('hi')\n"),
            stop_reason="tool_use",
        ),
        response(text("Done: created hello.py.")),
    ])
    result = build_agent(settings, llm=llm).run("create hello.py")

    assert result.status == "completed"
    assert result.final_text == "Done: created hello.py."
    assert result.steps == 2
    assert result.usage.input_tokens == 20
    assert (settings.workspace / "hello.py").read_text() == "print('hi')\n"

    # Second model call sees: system, task, assistant tool call, tool result
    second = llm.calls[1]
    assert [type(m) for m in second] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert second[-1].content == "Created hello.py (1 lines)." and second[-1].tool_call_id == "t1"


def test_tool_errors_go_back_to_model(settings):
    llm = FakeLLM([
        response(tool_use("t1", "read_file", path="missing.py"), stop_reason="tool_use"),
        response(text("That file does not exist.")),
    ])
    build_agent(settings, llm=llm).run("read missing.py")
    result = llm.calls[1][-1]
    assert result.status == "error" and "File not found" in result.content


def test_shell_commands_need_approval(settings):
    llm = FakeLLM([
        response(tool_use("t1", "run_command", command="echo hi"), stop_reason="tool_use"),
        response(text("ok")),
    ])
    seen = []
    agent = build_agent(settings, llm=llm, approve=lambda name, args: seen.append(args) or False)
    agent.run("say hi")
    assert seen == [{"command": "echo hi"}]
    assert "declined" in llm.calls[1][-1].content


def test_truncated_tool_call_is_answered_not_run(settings):
    llm = FakeLLM([
        response(tool_use("t1", "write_file", path="big.py", content="partial"), stop_reason="max_tokens"),
        response(text("Retried in smaller parts.")),
    ])
    result = build_agent(settings, llm=llm).run("write a big file")
    assert result.status == "completed"
    assert not (settings.workspace / "big.py").exists()
    assert llm.calls[1][-1].status == "error"


def test_step_limit(settings):
    loop = [response(tool_use(f"t{i}", "list_files", path="."), stop_reason="tool_use") for i in range(10)]
    result = build_agent(settings, llm=FakeLLM(loop)).run("loop forever")
    assert result.status == "max_steps"
    assert result.steps == settings.max_steps


def test_refusal_is_reported(settings):
    result = build_agent(settings, llm=FakeLLM([response(stop_reason="refusal")])).run("x")
    assert result.status == "refusal"


def test_sessions_are_checkpointed_and_resumable(settings):
    checkpointer = open_checkpointer(settings.database_url)
    first = build_agent(settings, llm=FakeLLM([response(text("first"))]), checkpointer=checkpointer).run("hello")
    assert first.session_id in list_sessions(checkpointer)

    llm = FakeLLM([response(text("second"))])
    agent = build_agent(settings, llm=llm, checkpointer=checkpointer)
    assert agent.session_exists(first.session_id)
    second = agent.run("again", session_id=first.session_id)

    assert second.session_id == first.session_id
    assert [type(m) for m in llm.calls[0]] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert second.steps == 1  # steps count per run


def test_transcript_groups_tool_results():
    messages = [
        HumanMessage("task"),
        AIMessage(content=[{"type": "text", "text": "hi"}],
                  tool_calls=[{"id": "a", "name": "x", "args": {}}, {"id": "b", "name": "y", "args": {}}]),
        ToolMessage(content="ok", tool_call_id="a"),
        ToolMessage(content="bad", tool_call_id="b", status="error"),
    ]
    out = to_transcript(messages)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][1] == {"type": "tool_use", "id": "a", "name": "x", "input": {}}
    assert out[2]["content"][1]["is_error"] is True
