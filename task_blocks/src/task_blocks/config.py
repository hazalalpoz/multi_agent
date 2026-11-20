from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_DIR = PROJECT_ROOT / "agent"
PROMPTS_DIR = AGENT_DIR / "prompts"
SKILLS_DIR = AGENT_DIR / "skills"
SOUL_PATH = AGENT_DIR / "SOUL.md"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelConfig:
    role: str
    model_id: str
    local: bool
    temperature: float
    supports_structured_output: bool


@dataclass(frozen=True)
class Settings:
    openrouter_base_url: str
    openrouter_api_key: str | None
    ollama_base_url: str

    router: ModelConfig
    verifier: ModelConfig
    therapist: ModelConfig
    logical: ModelConfig
    fallback: ModelConfig
    embedding_model: str

    history_max_messages: int
    max_reply_chars: int
    max_regeneration_attempts: int

    data_dir: Path


    log_level: str

    agent_dir: Path = field(default=AGENT_DIR)
    prompts_dir: Path = field(default=PROMPTS_DIR)
    skills_dir: Path = field(default=SKILLS_DIR)
    soul_path: Path = field(default=SOUL_PATH)

    @property
    def checkpointer_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def store_path(self) -> Path:
        return self.data_dir / "store.sqlite"

    def models_by_role(self) -> dict[str, ModelConfig]:
        return {
            "router": self.router,
            "verifier": self.verifier,
            "therapist": self.therapist,
            "logical": self.logical,
            "fallback": self.fallback,
        }


def _is_local_model(model_id: str) -> bool:
    return "/" not in model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openrouter_base_url=_env_str(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        router=ModelConfig(
            role="router",
            model_id=(_router_model := _env_str("ROUTER_MODEL", "qwen3:1.7b")),
            local=_is_local_model(_router_model),
            temperature=_env_float("ROUTER_TEMPERATURE", 0.0),
            supports_structured_output=True,
        ),
        verifier=ModelConfig(
            role="verifier",
            model_id=(_verifier_model := _env_str("VERIFIER_MODEL", "qwen3:1.7b")),
            local=_is_local_model(_verifier_model),
            temperature=_env_float("VERIFIER_TEMPERATURE", 0.0),
            supports_structured_output=True,
        ),
        therapist=ModelConfig(
            role="therapist",
            model_id=(_therapist_model := _env_str("THERAPIST_MODEL", "google/gemma-4-31b-it:free")),
            local=_is_local_model(_therapist_model),
            temperature=_env_float("THERAPIST_TEMPERATURE", 0.7),
            supports_structured_output=True,
        ),
        logical=ModelConfig(
            role="logical",
            model_id=(_logical_model := _env_str("LOGICAL_MODEL", "google/gemma-4-26b-a4b-it:free")),
            local=_is_local_model(_logical_model),
            temperature=_env_float("LOGICAL_TEMPERATURE", 0.2),
            supports_structured_output=True,
        ),
        fallback=ModelConfig(
            role="fallback",
            model_id=(
                _fallback_model := _env_str(
                    "FALLBACK_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
                )
            ),
            local=_is_local_model(_fallback_model),
            temperature=_env_float("FALLBACK_TEMPERATURE", 0.5),
            supports_structured_output=False,
        ),
        embedding_model=_env_str("EMBEDDING_MODEL", "nomic-embed-text"),
        history_max_messages=_env_int("HISTORY_MAX_MESSAGES", 12),
        max_reply_chars=_env_int("MAX_REPLY_CHARS", 4000),

        max_regeneration_attempts=_env_int("MAX_REGENERATION_ATTEMPTS", 1),
        data_dir=Path(_env_str("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser(),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
