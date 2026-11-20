from __future__ import annotations

import pytest

from task_blocks.tools import BASE_TOOLS, agent_tools, get_datetime, load_skill


def test_base_tools_are_registered() -> None:
    assert {tool.name for tool in BASE_TOOLS} == {"load_skill", "get_datetime"}


def test_every_tool_has_a_docstring_description() -> None:
    for tool in BASE_TOOLS:
        assert tool.description and len(tool.description) > 30


def test_load_skill_returns_the_body() -> None:
    result = load_skill.invoke({"name": "note-taking"})
    assert result.startswith("# Skill: note-taking")
    assert "Never save" in result


def test_load_skill_unknown_name_returns_error_text() -> None:
    result = load_skill.invoke({"name": "nope"})
    assert result.startswith("Error:")
    assert "note-taking" in result, "the error should list what is available"


def test_load_skill_empty_name_returns_error_text() -> None:
    assert load_skill.invoke({"name": ""}).startswith("Error:")
    assert load_skill.invoke({"name": "   "}).startswith("Error:")


def test_load_skill_never_raises() -> None:
    for bad in ["", "   ", "../../etc/passwd", "SKILL.md", "crisis response"]:
        assert isinstance(load_skill.invoke({"name": bad}), str)


def test_load_skill_does_not_traverse_paths() -> None:
    result = load_skill.invoke({"name": "../../../etc/passwd"})
    assert result.startswith("Error:")
    assert "root:" not in result


def test_get_datetime_defaults_to_utc() -> None:
    result = get_datetime.invoke({})
    assert "UTC" in result


def test_get_datetime_accepts_iana_zone() -> None:
    result = get_datetime.invoke({"timezone_name": "Europe/Moscow"})
    assert "Error" not in result
    assert any(day in result for day in
               ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"))


@pytest.mark.parametrize("bad", ["Moscow", "Mars/Olympus", "GMT+25", "!!"])
def test_get_datetime_rejects_bad_zone_with_guidance(bad: str) -> None:
    result = get_datetime.invoke({"timezone_name": bad})
    assert result.startswith("Error:")
    assert "IANA" in result


def test_get_datetime_handles_empty_zone() -> None:
    assert "UTC" in get_datetime.invoke({"timezone_name": ""})


def test_crisis_turn_has_no_memory_tools() -> None:
    names = {tool.name for tool in agent_tools(crisis=True)}
    assert names == {"load_skill", "get_datetime"}
    assert "manage_memory" not in names
    assert "search_memory" not in names


def test_non_crisis_turn_has_memory_tools() -> None:
    names = {tool.name for tool in agent_tools(crisis=False)}
    assert {"manage_memory", "search_memory"} <= names


def test_memory_tool_save_policy_comes_from_the_skill() -> None:
    from task_blocks.assets import get_skill
    from task_blocks.tools import _save_policy

    policy = _save_policy()
    assert get_skill("note-taking").body in policy
