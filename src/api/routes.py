"""FastAPI app exposing the LangGraph agent over HTTP.

There is no human at the keyboard here, so run_command is only allowed when
AGENT_AUTO_APPROVE=true. Leave it off unless the workspace is disposable.
"""

from __future__ import annotations

from typing import Any, Callable

import anthropic
from fastapi import FastAPI, HTTPException

from ..agent import build_agent, list_sessions, open_checkpointer, to_transcript, validate_session_id
from ..agent.state import Usage
from ..tools import ToolError, Workspace
from ..tools.filesystem import IGNORED_DIRS
from ..utils.config import Settings, load_settings
from .schemas import HealthResponse, RunRequest, RunResponse, UsageOut


def create_app(settings: Settings | None = None, llm_factory: Callable[[], Any] | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Coding Agent", version="1.0.0")
    checkpointer = open_checkpointer(settings.sessions_dir)
    workspace = Workspace(settings.workspace)

    def agent():
        return build_agent(settings, llm=llm_factory() if llm_factory else None, checkpointer=checkpointer)

    def check_session(session_id: str) -> None:
        try:
            validate_session_id(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(model=settings.model, workspace=str(settings.workspace))

    @app.get("/sessions")
    def sessions() -> dict[str, list[str]]:
        return {"sessions": list_sessions(checkpointer)}

    @app.get("/sessions/{session_id}")
    def session_detail(session_id: str) -> dict:
        check_session(session_id)
        messages = agent().messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")
        steps = sum(1 for m in to_transcript(messages) if m["role"] == "assistant")
        return {
            "session_id": session_id,
            "steps": steps,
            "usage": vars(Usage.from_messages(messages)),
            "messages": to_transcript(messages),
        }

    @app.get("/workspace/files")
    def workspace_files() -> dict[str, list[str]]:
        root = workspace.root
        files = [
            p.relative_to(root).as_posix()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != ".gitkeep"
            and not any(part in IGNORED_DIRS for part in p.relative_to(root).parts)
        ]
        return {"files": files}

    @app.get("/workspace/file")
    def workspace_file(path: str) -> dict[str, str]:
        try:
            target = workspace.resolve(path)
        except ToolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return {"path": path, "content": target.read_text(encoding="utf-8", errors="replace")}

    @app.post("/run", response_model=RunResponse)
    def run(request: RunRequest) -> RunResponse:
        runner = agent()
        if request.session_id:
            check_session(request.session_id)
            if not runner.session_exists(request.session_id):
                raise HTTPException(status_code=404, detail="Session not found")
        try:
            result = runner.run(request.task, session_id=request.session_id)
        except anthropic.RateLimitError as exc:
            raise HTTPException(status_code=429, detail="Rate limited by the model API; retry later.") from exc
        except anthropic.APIStatusError as exc:
            raise HTTPException(status_code=502, detail=f"Model API error ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise HTTPException(status_code=503, detail="Could not reach the model API.") from exc

        return RunResponse(
            status=result.status,
            final_text=result.final_text,
            steps=result.steps,
            session_id=result.session_id,
            usage=UsageOut(**vars(result.usage)),
        )

    return app
