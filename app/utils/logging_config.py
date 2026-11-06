from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "request_id", "model_id", "model_key", "bucket", "status"):
            value = getattr(record, key, None)
            if value is not None:
                event[key] = value
        return json.dumps(event, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise structured logs; callers must not pass input text or secrets."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
