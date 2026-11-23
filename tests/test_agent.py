from __future__ import annotations

from src.agent import AgentState, SessionMemory, build_agent
from tests.fakes import FakeLLM, response, text, tool_use


def test_agent_runs_tools_until_done(settings):
    llm = FakeLLM([
        response(
            text("Creating the file."),
            tool_use("t1", "write_file", path="hello.py", content="print('hi')\n"),
            stop_reason="tool_use",
        ),
        response(text("Done: created hello.py.")),
    ])
    agent = build_agent(settings, llm=llm)
    result = agent.run("create hello.py")

    assert result.status == "completed"
    assert result.final_text == "Done: created hello.py."
    assert result.steps == 2
    assert (settings.workspace / "hello.py").read_text() == "print('hi')\n"

    # Second call carries: user task, assistant tool_use, user tool_result
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"][0] == {
        "type": "tool_result", "tool_use_id": "t1", "content": "Created hello.py (1 lines)."
    }


def test_tool_errors_go_back_to_model(settings):
    llm = FakeLLM([
        response(tool_use("t1", "read_file", path="missing.py"), stop_reason="tool_use"),
        response(text("That file does not exist.")),
    ])
    build_agent(settings, llm=llm).run("read missing.py")
    result_block = llm.calls[1]["messages"][-1]["content"][0]
    assert result_block["is_error"] is True
    assert "File not found" in result_block["content"]


def test_shell_commands_need_approval(settings):
    llm = FakeLLM([
        response(tool_use("t1", "run_command", command="echo hi"), stop_reason="tool_use"),
        response(text("ok")),
    ])
    seen = []
    agent = build_agent(settings, llm=llm, approve=lambda name, inp: seen.append(inp) or False)
    agent.run("say hi")
    assert seen == [{"command": "echo hi"}]
    assert "declined" in llm.calls[1]["messages"][-1]["content"][0]["content"]


def test_truncated_tool_call_is_answered_not_run(settings):
    llm = FakeLLM([
        response(tool_use("t1", "write_file", path="big.py", content="partial"), stop_reason="max_tokens"),
        response(text("Retried in smaller parts.")),
    ])
    result = build_agent(settings, llm=llm).run("write a big file")
    assert result.status == "completed"
    assert not (settings.workspace / "big.py").exists()
    assert llm.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_step_limit(settings):
    loop = [response(tool_use(f"t{i}", "list_files", path="."), stop_reason="tool_use") for i in range(10)]
    result = build_agent(settings, llm=FakeLLM(loop)).run("loop forever")
    assert result.status == "max_steps"
    assert result.steps == settings.max_steps


def test_refusal_is_reported(settings):
    result = build_agent(settings, llm=FakeLLM([response(stop_reason="refusal")])).run("x")
    assert result.status == "refusal"


def test_session_is_saved_and_resumable(settings):
    agent = build_agent(settings, llm=FakeLLM([response(text("first"))]))
    first = agent.run("hello")
    loaded = SessionMemory(settings.sessions_dir).load(first.session_id)
    assert isinstance(loaded, AgentState)
    assert loaded.messages[1]["content"] == [{"type": "text", "text": "first"}]

    llm = FakeLLM([response(text("second"))])
    build_agent(settings, llm=llm).run("again", state=loaded)
    assert [m["role"] for m in llm.calls[0]["messages"]] == ["user", "assistant", "user"]
