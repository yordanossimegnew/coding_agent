from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import create_app
from tests.fakes import FakeLLM, response, text


def make_client(settings, responses):
    return TestClient(create_app(settings, llm_factory=lambda: FakeLLM(responses)))


def test_health(settings):
    body = make_client(settings, []).get("/health").json()
    assert body["status"] == "ok" and body["model"] == settings.model


def test_run_and_list_sessions(settings):
    client = make_client(settings, [response(text("All done."))])
    res = client.post("/run", json={"task": "do a thing"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed" and body["final_text"] == "All done."
    assert body["session_id"] in client.get("/sessions").json()["sessions"]


def test_unknown_session_404(settings):
    res = make_client(settings, []).post("/run", json={"task": "x", "session_id": "nope"})
    assert res.status_code == 404


def test_bad_session_id_400(settings):
    res = make_client(settings, []).post("/run", json={"task": "x", "session_id": "../etc"})
    assert res.status_code == 400


def test_session_detail_and_workspace_endpoints(settings):
    from tests.fakes import tool_use

    client = make_client(settings, [
        response(tool_use("t1", "write_file", path="a.py", content="x = 1\n"), stop_reason="tool_use"),
        response(text("wrote a.py")),
    ])
    sid = client.post("/run", json={"task": "write a.py"}).json()["session_id"]

    detail = client.get(f"/sessions/{sid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert client.get("/workspace/files").json()["files"] == ["a.py"]
    assert client.get("/workspace/file", params={"path": "a.py"}).json()["content"] == "x = 1\n"
    assert client.get("/workspace/file", params={"path": "../x"}).status_code == 400


def test_empty_task_rejected(settings):
    assert make_client(settings, []).post("/run", json={"task": ""}).status_code == 422
