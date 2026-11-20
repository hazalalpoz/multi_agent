from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from task_blocks.config import get_settings


class AssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


def _read(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AssetError(f"Missing agent asset: {path}") from exc
    except OSError as exc:
        raise AssetError(f"Could not read agent asset {path}: {exc}") from exc
    if not text.strip():
        raise AssetError(f"Agent asset is empty: {path}")
    return text


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    settings = get_settings()
    return _read(settings.prompts_dir / f"{name}.md").strip()


@lru_cache(maxsize=1)
def load_soul() -> str:
    return _read(get_settings().soul_path).strip()


def _parse_skill_file(path: Path) -> Skill:
    text = _read(path)
    if not text.startswith("---"):
        raise AssetError(f"Skill file has no YAML frontmatter: {path}")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise AssetError(f"Skill frontmatter is not terminated: {path}")

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise AssetError(f"Invalid YAML frontmatter in {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise AssetError(f"Skill frontmatter is not a mapping: {path}")

    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not name:
        raise AssetError(f"Skill frontmatter is missing 'name': {path}")
    if not description:
        raise AssetError(f"Skill frontmatter is missing 'description': {path}")
    if name != path.parent.name:
        raise AssetError(
            f"Skill name {name!r} does not match its directory {path.parent.name!r}: {path}"
        )

    return Skill(
        name=name, description=description, body=parts[2].strip(), path=path
    )


@lru_cache(maxsize=1)
def load_skills() -> dict[str, Skill]:
    skills_dir = get_settings().skills_dir
    if not skills_dir.is_dir():
        raise AssetError(f"Missing skills directory: {skills_dir}")

    skills: dict[str, Skill] = {}
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        skill = _parse_skill_file(skill_file)
        skills[skill.name] = skill

    if not skills:
        raise AssetError(f"No skills found under {skills_dir}")
    return skills


def get_skill(name: str) -> Skill | None:
    return load_skills().get(name.strip())


def skill_index() -> str:
    lines = [
        f"- `{skill.name}`: {skill.description}"
        for skill in load_skills().values()
    ]
    return "\n".join(lines)


def build_system_prompt(
    role: str,
    *,
    include_soul: bool = True,
    include_skills: bool = True,
    extra_sections: list[tuple[str, str]] | None = None,
) -> str:
    sections: list[str] = []
    if include_soul:
        sections.append(load_soul())
    sections.append(load_prompt(role))

    if include_skills:
        sections.append(
            "# Available skills\n\n"
            "Names and descriptions only. Call `load_skill(name)` for the full "
            "text of one.\n\n" + skill_index()
        )

    for title, body in extra_sections or []:
        if body and body.strip():
            sections.append(f"# {title}\n\n{body.strip()}")

    return "\n\n---\n\n".join(sections)
