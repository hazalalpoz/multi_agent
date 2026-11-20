"""Minimal Alertmanager webhook receiver.

Logs every alert it is sent, as one JSON line per alert, and keeps the most
recent ones in memory so `GET /alerts` can show what has fired. This is the
alert destination for the stack: self-contained, no external accounts, and
enough to demonstrate that the Prometheus → Alertmanager → destination path
actually works end to end.

Deliberately dependency-free — the standard library only — so the service needs
no image build and no requirements file.

Endpoints:
    POST /alerts   Alertmanager's webhook payload
    GET  /alerts   the alerts received so far, newest first
    GET  /health   liveness
"""

from __future__ import annotations

import json
import logging
import sys
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque

MAX_RETAINED = 200
PORT = 8080

_received: Deque[dict[str, Any]] = deque(maxlen=MAX_RETAINED)

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(message)s"
)
logger = logging.getLogger("alert-webhook")


def _log_alert(alert: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    labels = alert.get("labels") or {}
    annotations = alert.get("annotations") or {}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": alert.get("status") or group.get("status"),
        "alertname": labels.get("alertname"),
        "severity": labels.get("severity"),
        "summary": annotations.get("summary"),
        "description": (annotations.get("description") or "").strip(),
        "startsAt": alert.get("startsAt"),
        "endsAt": alert.get("endsAt"),
    }
    logger.info(json.dumps(record, ensure_ascii=False))
    return record


class Handler(BaseHTTPRequestHandler):
    server_version = "alert-webhook/1.0"

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") not in ("/alerts", ""):
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            group = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"bad payload: {exc}"})
            return

        alerts = group.get("alerts") or []
        if not isinstance(alerts, list):
            self._send_json(400, {"error": "alerts must be a list"})
            return

        for alert in alerts:
            if isinstance(alert, dict):
                _received.appendleft(_log_alert(alert, group))

        self._send_json(200, {"received": len(alerts)})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "ok", "retained": len(_received)})
        elif path in ("/alerts", ""):
            self._send_json(
                200, {"count": len(_received), "alerts": list(_received)}
            )
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Silence the default per-request access log; the alert lines are the
        # output that matters here.
        return


def main() -> None:
    logger.info(
        json.dumps({"event": "alert_webhook_started", "port": PORT})
    )
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
