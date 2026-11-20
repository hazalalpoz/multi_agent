from __future__ import annotations

import pytest

from task_blocks.assets import (
    AssetError,
    build_system_prompt,
    get_skill,
    load_prompt,
    load_skills,
    load_soul,
    skill_index,
)


REQUIRED_PROMPTS = ["router", "therapist", "logical", "verifier", "regenerate"]


REQUIRED_SKILLS = [
    "active-listening",
    "crisis-response",
    "structured-answer",
    "note-taking",
]


@pytest.mark.parametrize("name", REQUIRED_PROMPTS)
def test_every_required_prompt_loads(name: str) -> None:
    body = load_prompt(name)
    assert body.strip()
    assert len(body) > 200, f"prompt {name!r} is suspiciously short"


def test_missing_prompt_raises_asset_error() -> None:
    with pytest.raises(AssetError, match="Missing agent asset"):
        load_prompt("no-such-prompt")


def test_soul_loads() -> None:
    soul = load_soul()
    assert "never" in soul.lower()


def test_all_skills_load_and_are_well_formed() -> None:
    skills = load_skills()
    assert set(skills) == set(REQUIRED_SKILLS)
    for name, skill in skills.items():
        assert skill.name == name
        assert skill.description.strip(), f"{name} has no description"
        assert skill.body.strip(), f"{name} has an empty body"
        assert "---" not in skill.body.split("\n")[0]


def test_get_skill_returns_none_for_unknown() -> None:
    assert get_skill("no-such-skill") is None


def test_get_skill_tolerates_whitespace() -> None:
    assert get_skill("  note-taking  ") is not None


def test_skill_index_lists_names_and_descriptions_only() -> None:
    index = skill_index()
    for name in REQUIRED_SKILLS:
        assert f"`{name}`" in index

    crisis = get_skill("crisis-response")
    assert crisis is not None

    assert "Never minimise" not in index
    assert len(index) < len(crisis.body)


def test_crisis_skill_contains_only_verified_resources() -> None:
    body = get_skill("crisis-response").body
    assert "112" in body
    assert "findahelpline.com" in body
    for rejected in ("8-800-2000-122", "8-800-555-22-20", "222-00-24"):
        assert rejected not in body, f"unverified resource {rejected} is back"


def test_build_system_prompt_orders_soul_before_role() -> None:
    prompt = build_system_prompt("therapist")
    assert prompt.index(load_soul()[:60]) < prompt.index(load_prompt("therapist")[:60])


def test_build_system_prompt_includes_skill_index_by_default() -> None:
    assert "# Available skills" in build_system_prompt("logical")


def test_build_system_prompt_can_omit_skills() -> None:
    prompt = build_system_prompt("regenerate", include_skills=False)
    assert "# Available skills" not in prompt


def test_build_system_prompt_adds_extra_sections() -> None:
    prompt = build_system_prompt(
        "logical", extra_sections=[("Retrieved memories", "- Likes short answers")]
    )
    assert "# Retrieved memories" in prompt
    assert "- Likes short answers" in prompt


def test_build_system_prompt_skips_empty_extra_sections() -> None:
    prompt = build_system_prompt("logical", extra_sections=[("Empty", "   ")])
    assert "# Empty" not in prompt


def test_no_prompt_text_is_hardcoded_in_python() -> None:
    from pathlib import Path

    from task_blocks.config import PROJECT_ROOT

    sources = {
        path: path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "src" / "task_blocks").glob("*.py")
    }
    for name in REQUIRED_PROMPTS:
        first_line = load_prompt(name).splitlines()[0].strip()
        assert len(first_line) > 20
        for path, source in sources.items():
            assert first_line not in source, (
                f"prompt {name!r} appears inlined in {Path(path).name}"
            )
