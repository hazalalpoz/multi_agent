from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        from task_blocks.observability import current_trace_id

        trace_id = current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None) -> None:
    from task_blocks.config import get_settings

    resolved = (level or get_settings().log_level).upper()
    root = logging.getLogger()
    root.setLevel(resolved)

    for handler in root.handlers:
        if getattr(handler, "_task_blocks", False):
            handler.setLevel(resolved)
            return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(resolved)
    handler._task_blocks = True
    root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
