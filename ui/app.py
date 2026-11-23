"""Streamlit front end for the Tiny Coding Agent. Talks to the FastAPI backend over HTTP.

    uvicorn main:app            # terminal 1
    streamlit run ui/app.py     # terminal 2
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")
EXAMPLES = json.loads((Path(__file__).resolve().parents[1] / "data" / "examples.json").read_text())

st.set_page_config(page_title="Tiny Coding Agent", page_icon="🛠️", layout="wide")


# ----------------------------------------------------------------------------- API helpers
def api_get(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_run(task: str, session_id: str | None):
    r = requests.post(f"{API_URL}/run", json={"task": task, "session_id": session_id}, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json",
        ".md": "markdown", ".toml": "toml", ".yml": "yaml", ".yaml": "yaml", ".sh": "bash"}


def format_call(name: str, args: dict) -> str:
    """Render a tool call like write_file(path="a.py", content=<12 lines>)."""
    parts = []
    for key, value in args.items():
        value = str(value)
        shown = f"<{len(value.splitlines())} lines>" if "\n" in value else json.dumps(value[:80])
        parts.append(f"{key}={shown}")
    return f"{name}({', '.join(parts)})"


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛠️ Tiny Coding Agent")
    st.caption("A minimal agent loop on the Claude API")
    health = api_get("/health")
    if health:
        st.success("API connected", icon="✅")
        st.markdown(f"**Model:** `{health['model']}`")
        st.markdown(f"**Workspace:** `{Path(health['workspace']).name}/`")
    else:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn main:app`.", icon="🚫")

    st.divider()
    sessions = (api_get("/sessions") or {}).get("sessions", [])
    choice = st.selectbox("Session", ["➕ New session", *reversed(sessions)])
    session_id = None if choice.startswith("➕") else choice
    st.caption("Choose a past session to continue it or view its transcript.")


# ----------------------------------------------------------------------------- main
tab_run, tab_trace, tab_files = st.tabs(["▶️ Run a task", "🧭 Agent trace", "📁 Workspace"])

with tab_run:
    st.subheader("What should the agent build?")
    example = st.selectbox("Start from an example", ["—", *[e["name"] for e in EXAMPLES]])
    default = next((e["task"] for e in EXAMPLES if e["name"] == example), "")
    task = st.text_area("Task", value=default, height=120,
                        placeholder="e.g. Create fizzbuzz.py with a fizzbuzz(n) function and pytest tests.")
    if st.button("Run agent", type="primary", disabled=not (task.strip() and health)):
        with st.spinner("Agent is working… (reading, writing and running code)"):
            try:
                st.session_state["last"] = api_run(task, session_id)
            except (RuntimeError, requests.RequestException) as exc:
                st.error(f"Run failed: {exc}")

    last = st.session_state.get("last")
    if last:
        badge = {"completed": "🟢", "refusal": "🟠", "max_steps": "🟡"}.get(last["status"], "🔴")
        st.markdown(f"### {badge} {last['status'].replace('_', ' ').title()}")
        u = last["usage"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", last["steps"])
        c2.metric("Input tokens", f"{u['input_tokens']:,}")
        c3.metric("Output tokens", f"{u['output_tokens']:,}")
        c4.metric("Cached tokens", f"{u['cache_read_input_tokens']:,}")
        st.markdown(last["final_text"] or "_(no text)_")
        st.caption(f"Session `{last['session_id']}`. Pick it in the sidebar to continue it.")

with tab_trace:
    sid = session_id or (st.session_state.get("last") or {}).get("session_id")
    detail = api_get(f"/sessions/{sid}") if sid else None
    if not detail:
        st.info("Run a task or pick a session in the sidebar to see each step the agent took.")
    else:
        st.caption(f"Session `{detail['session_id']}`: {detail['steps']} model calls")
        for msg in detail["messages"]:
            content = msg["content"]
            if isinstance(content, str):
                with st.chat_message("user"):
                    st.markdown(content)
                continue
            for block in content:
                kind = block.get("type")
                if kind == "text" and block.get("text"):
                    with st.chat_message("assistant"):
                        st.markdown(block["text"])
                elif kind == "tool_use":
                    with st.chat_message("assistant", avatar="🔧"):
                        st.code(format_call(block["name"], block["input"]), language="python")
                elif kind == "tool_result":
                    content_text = str(block.get("content", ""))
                    first_line = content_text.splitlines()[0][:90] if content_text else ""
                    icon = "⚠️" if block.get("is_error") else "📤"
                    with st.expander(f"{icon} {first_line}", expanded=False):
                        st.code(content_text[:4000])

with tab_files:
    files = (api_get("/workspace/files") or {}).get("files", [])
    if not files:
        st.info("The workspace is empty. Files the agent creates will appear here.")
    else:
        left, right = st.columns([1, 3])
        picked = left.radio("Files", files, label_visibility="collapsed")
        data = api_get("/workspace/file", {"path": picked})
        if data:
            right.markdown(f"**`{picked}`**")
            right.code(data["content"], language=LANG.get(Path(picked).suffix, None))
