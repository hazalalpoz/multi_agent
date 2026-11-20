from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


_LLM_BUCKETS = (0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
_REQUEST_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0)

agent_requests_total = Counter(
    "agent_requests_total",
    "Completed agent turns.",
    ["route", "outcome"],
)

agent_request_latency_seconds = Histogram(
    "agent_request_latency_seconds",
    "End-to-end latency of one agent turn.",
    ["route"],
    buckets=_REQUEST_BUCKETS,
)

llm_calls_total = Counter(
    "llm_calls_total",
    "LLM calls, by model, role and outcome.",
    ["model", "role", "status"],
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "Latency of a single LLM call.",
    ["model", "role"],
    buckets=_LLM_BUCKETS,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Tokens consumed, by model and kind (prompt/completion).",
    ["model", "type"],
)

fallback_total = Counter(
    "fallback_total",
    "Fallbacks taken, by reason.",
    ["reason"],
)

verifier_verdicts_total = Counter(
    "verifier_verdicts_total",
    "Verifier verdicts, by result and deciding stage.",
    ["result", "source"],
)

tool_calls_total = Counter(
    "tool_calls_total",
    "Tool invocations, by tool and outcome.",
    ["tool", "status"],
)

crisis_detected_total = Counter(
    "crisis_detected_total",
    "Turns flagged as a crisis, by what detected it.",
    ["detector"],
)

memory_ops_total = Counter(
    "memory_ops_total",
    "Long-term memory operations, by operation and outcome.",
    ["op", "status"],
)

router_confidence = Histogram(
    "router_confidence",
    "Router self-reported confidence in its route.",
    buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
)

build_info = Gauge(
    "task_blocks_build_info",
    "Static build and configuration info, as labels.",
    ["version", "router_model", "therapist_model", "logical_model", "fallback_model"],
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def set_build_info() -> None:
    from task_blocks import __version__
    from task_blocks.config import get_settings

    settings = get_settings()
    build_info.labels(
        version=__version__,
        router_model=settings.router.model_id,
        therapist_model=settings.therapist.model_id,
        logical_model=settings.logical.model_id,
        fallback_model=settings.fallback.model_id,
    ).set(1)


@contextmanager
def track_llm_call(model: str, role: str) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    result: dict[str, Any] = {}
    status = "ok"
    try:
        yield result
    except Exception:
        status = "error"
        raise
    finally:
        llm_latency_seconds.labels(model=model, role=role).observe(
            time.perf_counter() - started
        )
        llm_calls_total.labels(model=model, role=role, status=status).inc()
        for kind in ("prompt", "completion"):
            count = result.get(f"{kind}_tokens")
            if isinstance(count, int) and count > 0:
                llm_tokens_total.labels(model=model, type=kind).inc(count)


def token_usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return {}
    counts: dict[str, int] = {}
    for key, kind in (("input_tokens", "prompt"), ("output_tokens", "completion")):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            counts[f"{kind}_tokens"] = value
    return counts


_tracing_ready = False


def _probe_collector(endpoint: str) -> None:
    try:
        import httpx

        response = httpx.post(
            f"{endpoint.rstrip('/')}/v1/traces",
            content=b"",
            headers={"content-type": "application/x-protobuf"},
            timeout=3.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tracing_collector_unreachable",
            extra={"endpoint": endpoint, "error": f"{type(exc).__name__}: {exc}"},
        )
        return

    if response.status_code in (401, 403):
        logger.error(
            "tracing_collector_rejects_auth",
            extra={
                "endpoint": endpoint,
                "status": response.status_code,
                "hint": (
                    "the collector requires authentication; set PHOENIX_API_KEY "
                    "or disable auth on the collector. Spans are being dropped."
                ),
            },
        )
    else:
        logger.info(
            "tracing_collector_reachable",
            extra={"endpoint": endpoint, "status": response.status_code},
        )


def configure_tracing() -> bool:
    global _tracing_ready
    if _tracing_ready:
        return True

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if not endpoint:
        logger.info("tracing_disabled", extra={"reason": "PHOENIX_COLLECTOR_ENDPOINT unset"})
        return False

    _probe_collector(endpoint)

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        api_key = os.getenv("PHOENIX_API_KEY", "").strip()
        if api_key:
            existing = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "")
            header = f"authorization=Bearer {api_key}"
            os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
                f"{existing},{header}" if existing else header
            )

        tracer_provider = register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "task-blocks"),
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
            batch=True,
            set_global_tracer_provider=True,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        _tracing_ready = True
        logger.info("tracing_enabled", extra={"endpoint": endpoint})
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tracing_setup_failed",
            extra={"error": f"{type(exc).__name__}: {exc}", "endpoint": endpoint},
        )
        return False


def current_trace_id() -> str | None:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        context = span.get_span_context()
        if context is None or not context.trace_id:
            return None
        return format(context.trace_id, "032x")
    except Exception:  # noqa: BLE001
        return None


def set_span_attributes(**attributes: Any) -> None:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"task_blocks.{key}", value)
    except Exception:
        pass
