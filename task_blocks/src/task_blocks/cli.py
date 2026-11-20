from __future__ import annotations

import argparse
import logging
import sys
import uuid

from task_blocks.config import get_settings
from task_blocks.graph import build_graph, reply_text, run_turn
from task_blocks.logging_config import configure_logging
from task_blocks.memory import memory_namespace, get_store
from task_blocks.observability import configure_tracing, set_build_info

logger = logging.getLogger(__name__)

BANNER = """task-blocks — emotional-support and factual assistant
Type a message, or /exit to quit, /help for commands.
"""

HELP = """Commands:
  /exit, /quit   leave
  /help          this list
  /info          route, model and verdict for the last turn
  /models        the configured routing table
  /memories      long-term memories stored for this user
"""


def _print_models() -> None:
    settings = get_settings()
    print("\nRouting table:")
    for role, cfg in settings.models_by_role().items():
        engine = "local Ollama" if cfg.local else "OpenRouter"
        print(f"  {role:<10} {cfg.model_id:<42} {engine}")
    print(f"  {'embeddings':<10} {settings.embedding_model:<42} local Ollama\n")


def _print_memories(user_id: str) -> None:
    try:
        items = get_store().search(memory_namespace(user_id), limit=50)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[error] could not read memories: {type(exc).__name__}: {exc}\n")
        return
    if not items:
        print(f"\nNo long-term memories stored for {user_id!r}.\n")
        return
    print(f"\nLong-term memories for {user_id!r}:")
    for item in items:
        value = item.value
        content = value.get("content", value) if isinstance(value, dict) else value
        print(f"  - {content}")
    print()


def _print_info(last: dict | None) -> None:
    if not last:
        print("\nNo turn yet.\n")
        return
    verdict = last.get("verdict") or {}
    print(
        f"\n  route       {last.get('route')}"
        f"\n  crisis      {last.get('crisis')}"
        f"\n  confidence  {last.get('confidence')}"
        f"\n  model_used  {last.get('model_used')}"
        f"\n  attempts    {last.get('attempts')}"
        f"\n  memories    {len(last.get('memories') or [])} retrieved"
        f"\n  verdict     {'pass' if verdict.get('passed') else 'fail'}"
        f" (via {verdict.get('source', 'n/a')})"
    )
    for reason in verdict.get("reasons") or []:
        print(f"              - {reason}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="task-blocks", description="Chat with the multi-agent assistant."
    )
    parser.add_argument(
        "--user",
        default="local-user",
        help="User ID; scopes long-term memory across threads.",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread ID to resume. Default: a new random thread.",
    )
    parser.add_argument("--log-level", default=None, help="Override LOG_LEVEL.")
    args = parser.parse_args()

    configure_logging(args.log_level)
    configure_tracing()
    set_build_info()
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"

    try:
        graph = build_graph()
    except Exception as exc:  # noqa: BLE001
        print(f"[fatal] could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    last_state: dict | None = None

    print(BANNER)
    print(f"user={args.user}  thread={thread_id}")
    print(f"(resume this conversation later with --thread {thread_id})\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue
        command = user_input.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            print("Bye.")
            return
        if command == "/help":
            print(HELP)
            continue
        if command == "/models":
            _print_models()
            continue
        if command == "/info":
            _print_info(last_state)
            continue
        if command == "/memories":
            _print_memories(args.user)
            continue

        try:
            last_state = run_turn(
                graph, user_input, user_id=args.user, thread_id=thread_id
            )
        except Exception as exc:
            logger.exception("cli_turn_failed")
            print(f"\n[error] {type(exc).__name__}: {exc}\n", file=sys.stderr)
            continue

        print(f"\nAssistant: {reply_text(last_state)}\n")


if __name__ == "__main__":
    main()
