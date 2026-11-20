from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool
from langmem import create_manage_memory_tool, create_search_memory_tool

from task_blocks.assets import AssetError, get_skill, load_skills
from task_blocks.memory import MEMORY_NAMESPACE, get_store
from task_blocks.observability import memory_ops_total, tool_calls_total

logger = logging.getLogger(__name__)


@tool(parse_docstring=True)
def load_skill(name: str) -> str:
    """Load the full text of one of the available skills.

    Your system prompt lists each skill by name and description. Call this to
    get the complete instructions for one of them before applying it.

    Args:
        name: The exact skill name, for example "active-listening".
    """
    requested = (name or "").strip()
    if not requested:
        tool_calls_total.labels(tool="load_skill", status="invalid_args").inc()
        return "Error: no skill name given. Pass the exact name of a skill."

    try:
        skill = get_skill(requested)
        known = sorted(load_skills())
    except AssetError as exc:
        tool_calls_total.labels(tool="load_skill", status="error").inc()
        logger.warning("load_skill failed: %s", exc)
        return f"Error: skills could not be loaded ({exc})."

    if skill is None:
        tool_calls_total.labels(tool="load_skill", status="not_found").inc()
        return (
            f"Error: there is no skill called {requested!r}. "
            f"Available skills: {', '.join(known)}."
        )

    tool_calls_total.labels(tool="load_skill", status="ok").inc()
    logger.info("tool=load_skill skill=%s", skill.name)
    return f"# Skill: {skill.name}\n\n{skill.body}"


@tool(parse_docstring=True)
def get_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time.

    Use this whenever the answer depends on what day or time it is, rather than
    assuming a date.

    Args:
        timezone_name: IANA timezone name, for example "Europe/Moscow".
            Defaults to UTC.
    """
    requested = (timezone_name or "UTC").strip() or "UTC"
    try:
        tz = timezone.utc if requested.upper() == "UTC" else ZoneInfo(requested)
    except (ZoneInfoNotFoundError, ValueError):
        tool_calls_total.labels(tool="get_datetime", status="invalid_args").inc()
        return (
            f"Error: unknown timezone {requested!r}. Use an IANA name such as "
            '"Europe/Moscow", or "UTC".'
        )

    now = datetime.now(tz)
    tool_calls_total.labels(tool="get_datetime", status="ok").inc()
    logger.info("tool=get_datetime tz=%s", requested)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


BASE_TOOLS: list[BaseTool] = [load_skill, get_datetime]


def _save_policy() -> str:
    skill = None
    try:
        skill = get_skill("note-taking")
    except AssetError as exc:
        logger.warning("note_taking_skill_unavailable", extra={"error": str(exc)})

    if skill is not None:
        return (
            "Save a durable fact about this person so it is available in future "
            "conversations. Follow this policy exactly; the default is not to "
            "save.\n\n" + skill.body
        )

    return (
        "Save only durable facts about this person: stated preferences, ongoing "
        "situations, things they asked you to remember. Never save transient "
        "feelings, identifiers, financial or medical data, credentials, third "
        "parties' private information, or anything from a conversation "
        "involving suicide, self-harm or immediate danger. One fact per memory."
    )


def _count_memory_op(tool: BaseTool, op: str) -> BaseTool:
    original = getattr(tool, "func", None)
    original_async = getattr(tool, "coroutine", None)

    if callable(original):
        def counted(*args: object, **kwargs: object) -> object:
            try:
                result = original(*args, **kwargs)
            except Exception:
                memory_ops_total.labels(op=op, status="error").inc()
                tool_calls_total.labels(tool=tool.name, status="error").inc()
                raise
            memory_ops_total.labels(op=op, status="ok").inc()
            tool_calls_total.labels(tool=tool.name, status="ok").inc()
            return result

        tool.func = counted

    if callable(original_async):
        async def counted_async(*args: object, **kwargs: object) -> object:
            try:
                result = await original_async(*args, **kwargs)
            except Exception:
                memory_ops_total.labels(op=op, status="error").inc()
                tool_calls_total.labels(tool=tool.name, status="error").inc()
                raise
            memory_ops_total.labels(op=op, status="ok").inc()
            tool_calls_total.labels(tool=tool.name, status="ok").inc()
            return result

        tool.coroutine = counted_async

    return tool


def memory_tools() -> list[BaseTool]:
    namespace = (MEMORY_NAMESPACE, "{user_id}")
    store = get_store()
    return [
        _count_memory_op(
            create_search_memory_tool(namespace=namespace, store=store),
            "search_tool",
        ),
        _count_memory_op(
            create_manage_memory_tool(
                namespace=namespace, store=store, instructions=_save_policy()
            ),
            "manage",
        ),
    ]


def agent_tools(*, crisis: bool = False) -> list[BaseTool]:
    if crisis:
        return list(BASE_TOOLS)

    try:
        return [*BASE_TOOLS, *memory_tools()]
    except Exception as exc:
        logger.error(
            "memory_tools_unavailable",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return list(BASE_TOOLS)
