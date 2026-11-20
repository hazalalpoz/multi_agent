from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from task_blocks import memory
from task_blocks.config import get_settings
from task_blocks.graph import build_graph, reply_text, run_turn
from task_blocks.logging_config import configure_logging
from task_blocks.observability import (
    configure_tracing,
    current_trace_id,
    render_metrics,
    set_build_info,
)

logger = logging.getLogger(__name__)

_graph: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    configure_logging()
    settings = get_settings()
    tracing = configure_tracing()
    set_build_info()
    _graph = build_graph()
    logger.info(
        "api_started",
        extra={
            "ollama": settings.ollama_base_url,
            "data_dir": str(settings.data_dir),
            "tracing": tracing,
        },
    )
    yield
    _graph = None


app = FastAPI(
    title="task-blocks",
    version="0.1.0",
    description=(
        "Multi-agent assistant: routing between an emotional-support agent and "
        "a factual agent, with a verification layer and long-term memory."
    ),
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=128,
        description="Scopes long-term memory. Stable per person.",
    )
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        description="Conversation to continue. Omit to start a new one.",
    )
    message: str = Field(
        min_length=1, max_length=8000, description="The user's message."
    )


class VerdictModel(BaseModel):
    passed: bool
    reasons: list[str] = []
    source: str = "none"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    trace_id: str | None = None
    route: str | None = None
    crisis: bool = False
    confidence: float | None = None
    model_used: str | None = None
    attempts: int = 0
    memories_used: int = 0
    verdict: VerdictModel
    latency_ms: int


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph is not ready.")

    thread_id = request.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()

    try:
        state = run_turn(
            _graph, request.message, user_id=request.user_id, thread_id=thread_id
        )
    except Exception as exc:
        logger.exception("chat_failed", extra={"thread_id": thread_id})
        raise HTTPException(
            status_code=502, detail=f"{type(exc).__name__}: {exc}"
        ) from exc

    verdict = state.get("verdict") or {}
    latency_ms = int((time.perf_counter() - started) * 1000)
    trace_id = current_trace_id()

    logger.info(
        "chat_turn",
        extra={
            "thread_id": thread_id,
            "route": state.get("route"),
            "crisis": bool(state.get("crisis")),
            "model": state.get("model_used"),
            "attempts": state.get("attempts", 0),
            "verdict_passed": bool(verdict.get("passed")),
            "latency_ms": latency_ms,
        },
    )

    return ChatResponse(
        reply=reply_text(state),
        thread_id=thread_id,
        trace_id=trace_id,
        route=state.get("route"),
        crisis=bool(state.get("crisis")),
        confidence=state.get("confidence"),
        model_used=state.get("model_used"),
        attempts=int(state.get("attempts", 0)),
        memories_used=len(state.get("memories") or []),
        verdict=VerdictModel(
            passed=bool(verdict.get("passed")),
            reasons=list(verdict.get("reasons") or []),
            source=str(verdict.get("source", "none")),
        ),
        latency_ms=latency_ms,
    )


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


def _ollama_health() -> dict[str, Any]:
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        installed = {
            model.get("name", "") for model in response.json().get("models", [])
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    required = {
        settings.router.model_id,
        settings.verifier.model_id,
        settings.embedding_model,
    }
    normalised = {name.removesuffix(":latest") for name in installed}
    missing = sorted(
        name for name in required if name.removesuffix(":latest") not in normalised
    )
    return {
        "status": "ok" if not missing else "degraded",
        "missing_models": missing,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    ollama = _ollama_health()
    storage = memory.health()

    healthy = (
        _graph is not None
        and ollama["status"] == "ok"
        and storage["checkpointer"] == "ok"
        and storage["store"] == "ok"
        and bool(storage.get("vector_search"))
    )
    return {
        "status": "ok" if healthy else "degraded",
        "graph": "ready" if _graph is not None else "not_ready",
        "ollama": ollama,
        "storage": storage,
        "models": {
            role: cfg.model_id for role, cfg in settings.models_by_role().items()
        },
    }
