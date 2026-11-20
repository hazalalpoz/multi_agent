from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from task_blocks.config import ModelConfig, Settings, get_settings

logger = logging.getLogger(__name__)


class ModelConfigurationError(RuntimeError):
    pass


def _build_local(cfg: ModelConfig, settings: Settings) -> BaseChatModel:
    return ChatOllama(
        model=cfg.model_id,
        base_url=settings.ollama_base_url,
        temperature=cfg.temperature,
        reasoning=False,
    )


def _build_remote(cfg: ModelConfig, settings: Settings) -> BaseChatModel:
    if not settings.openrouter_api_key:
        raise ModelConfigurationError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
            "the key, or override the model with a local one."
        )
    return ChatOpenAI(
        model=cfg.model_id,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=cfg.temperature,
        timeout=120,
        max_retries=2,
    )


def _build(cfg: ModelConfig, settings: Settings) -> BaseChatModel:
    return (
        _build_local(cfg, settings)
        if cfg.local
        else _build_remote(cfg, settings)
    )


@lru_cache(maxsize=None)
def get_model(role: str, *, with_transport_fallback: bool = True) -> BaseChatModel:
    settings = get_settings()
    table = settings.models_by_role()
    if role not in table:
        raise ModelConfigurationError(
            f"Unknown model role {role!r}. Known roles: {sorted(table)}"
        )

    cfg = table[role]
    model = _build(cfg, settings)

    if with_transport_fallback and not cfg.local and role != "fallback":
        fallback_cfg = settings.fallback
        if fallback_cfg.model_id != cfg.model_id:
            model = model.with_fallbacks(
                [_build(fallback_cfg, settings)],
                exceptions_to_handle=(Exception,),
            )
            logger.debug(
                "role=%s primary=%s transport_fallback=%s",
                role,
                cfg.model_id,
                fallback_cfg.model_id,
            )

    return model


def model_id_for(role: str) -> str:
    settings = get_settings()
    table = settings.models_by_role()
    if role not in table:
        raise ModelConfigurationError(f"Unknown model role {role!r}")
    return table[role].model_id


def reset_model_cache() -> None:
    get_model.cache_clear()
