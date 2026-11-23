"""Entry point. The project is a FastAPI application; a terminal CLI is included for local use.

    uvicorn main:app --reload                           # API server (docs at /docs)
    python main.py                                      # same, on port 8000
    python main.py cli "add fizzbuzz.py with tests"     # one task in the terminal
    python main.py cli                                  # interactive terminal chat
    python main.py cli --session abc123 "now add docs"  # resume a saved session
"""

from __future__ import annotations

import argparse
import json
import sys

import anthropic

from src.agent import AgentState, SessionMemory, build_agent
from src.api import create_app
from src.utils import load_settings, setup_logging

settings = load_settings()
setup_logging(settings.log_level, settings.log_dir)
app = create_app(settings)

DIM, BOLD, CYAN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


# ----------------------------------------------------------------------------- CLI
def print_event(kind: str, payload) -> None:
    if kind == "text":
        print(f"\n{payload}")
    elif kind == "tool_call":
        args = json.dumps(payload.input, ensure_ascii=False)
        print(f"{CYAN}> {payload.name}{RESET} {DIM}{args[:160]}{RESET}")
    elif kind == "tool_result" and payload.get("is_error"):
        print(f"{RED}  ! {str(payload['content'])[:200]}{RESET}")


def ask_approval(name: str, tool_input: dict) -> bool:
    print(f"\n{YELLOW}The agent wants to run:{RESET} {BOLD}{tool_input.get('command')}{RESET}")
    try:
        return input("Allow? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def run_task(agent, task: str, state: AgentState | None) -> AgentState:
    try:
        result = agent.run(task, state=state)
    except anthropic.AuthenticationError:
        sys.exit(f"{RED}Authentication failed. Set ANTHROPIC_API_KEY in .env (see .env.example).{RESET}")
    except anthropic.APIError as exc:
        print(f"{RED}API error: {exc}{RESET}")
        return state or AgentState()
    u = result.usage
    print(
        f"\n{DIM}[{result.status}] steps={result.steps} session={result.session_id} "
        f"tokens in={u.input_tokens} out={u.output_tokens} cached={u.cache_read_input_tokens}{RESET}"
    )
    return agent.memory.load(result.session_id)


def cli(args: argparse.Namespace) -> None:
    cfg = load_settings(workspace=args.workspace, auto_approve=True if args.yes else None)
    approve = (lambda *_: True) if cfg.auto_approve else ask_approval
    agent = build_agent(cfg, approve=approve, on_event=print_event)
    print(f"{DIM}workspace: {cfg.workspace}  model: {cfg.model}{RESET}")

    state = None
    if args.session:
        state = SessionMemory(cfg.sessions_dir).load(args.session)
        if state is None:
            sys.exit(f"No saved session '{args.session}'.")

    if args.task:
        run_task(agent, args.task, state)
        return

    print("Interactive mode. Type a task, or 'exit' to quit.")
    while True:
        try:
            task = input(f"\n{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task.lower() in {"exit", "quit"}:
            break
        if task:
            state = run_task(agent, task, state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding Agent")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the FastAPI server (default).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    c = sub.add_parser("cli", help="Use the agent from the terminal.")
    c.add_argument("task", nargs="?", help="Task to run. Omit for interactive mode.")
    c.add_argument("--workspace", help="Directory the agent works in (default: ./workspace).")
    c.add_argument("--session", help="Resume a saved session id.")
    c.add_argument("--yes", action="store_true", help="Auto-approve shell commands (careful!).")

    args = parser.parse_args()
    if args.command == "cli":
        cli(args)
    else:
        import uvicorn

        uvicorn.run(app, host=getattr(args, "host", "0.0.0.0"), port=getattr(args, "port", 8000))


if __name__ == "__main__":
    main()
