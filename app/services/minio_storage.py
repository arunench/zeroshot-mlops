from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any

from minio import Minio

logger = logging.getLogger(__name__)
REQUIRED_FIELDS = {
    "request_id",
    "timestamp",
    "input_text",
    "predicted_class",
    "confidence",
    "candidate_labels",
    "ranked_predictions",
    "model_name",
    "model_id",
    "application_version",
}


class StorageError(RuntimeError):
    """Raised when a prediction cannot be validated or persisted."""


def create_request_id() -> str:
    """Create a collision-resistant identifier for one prediction."""
    return str(uuid.uuid4())


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise StorageError("Prediction timestamp must be an ISO 8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StorageError("Prediction timestamp must be an ISO 8601 UTC string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise StorageError("Prediction timestamp must use UTC")
    return parsed.astimezone(UTC)


def _score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StorageError(f"Prediction {field_name} must be a number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise StorageError(f"Prediction {field_name} must be a number in [0, 1]")
    return normalized


def _validated_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], datetime]:
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise StorageError(f"Prediction is missing required fields: {', '.join(sorted(missing))}")
    request_id = record["request_id"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise StorageError("Prediction request_id must be a non-empty unique string")
    timestamp = _parse_timestamp(record["timestamp"])
    labels = record["candidate_labels"]
    ranked = record["ranked_predictions"]
    if not isinstance(labels, list) or len(labels) < 2:
        raise StorageError("Prediction candidate_labels must contain at least two labels")
    if not isinstance(ranked, list) or len(ranked) != len(labels):
        raise StorageError("Prediction ranked_predictions must include every candidate label")

    normalized_scores: list[float] = []
    normalized_ranked: list[dict[str, Any]] = []
    for item in ranked:
        if not isinstance(item, Mapping) or not isinstance(item.get("label"), str):
            raise StorageError("Each ranked prediction must contain a label and score")
        score = _score(item.get("score"), "ranked score")
        normalized_scores.append(score)
        normalized_ranked.append({"label": item["label"], "score": score})
    if normalized_scores != sorted(normalized_scores, reverse=True):
        raise StorageError("Ranked predictions must be sorted by descending score")

    confidence = _score(record["confidence"], "confidence")
    if confidence != normalized_scores[0]:
        raise StorageError("Prediction confidence must match the highest ranked score")
    cleaned = dict(record)
    cleaned["ranked_predictions"] = normalized_ranked
    if cleaned.get("model_revision") is None:
        cleaned.pop("model_revision", None)
    return cleaned, timestamp


class MinIOStorage:
    """Create the configured bucket and persist validated prediction JSON records."""

    def __init__(self, settings: Any, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def ensure_bucket(self) -> None:
        bucket = self._settings.minio_bucket
        try:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info(
                    "Created MinIO bucket",
                    extra={"event": "minio_bucket_create", "bucket": bucket},
                )
        except Exception as exc:
            raise StorageError(f"Unable to ensure MinIO bucket '{bucket}'") from exc

    def write_prediction(self, record: Mapping[str, Any]) -> str:
        validated, timestamp = _validated_record(record)
        bucket = self._settings.minio_bucket
        object_name = f"predictions/{timestamp:%Y/%m/%d}/{validated['request_id']}.json"
        payload = json.dumps(validated, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            self.ensure_bucket()
            self._client.put_object(
                bucket,
                object_name,
                BytesIO(payload),
                length=len(payload),
                content_type="application/json",
            )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError("Unable to persist prediction to MinIO") from exc
        logger.info(
            "Persisted prediction to MinIO",
            extra={
                "event": "minio_prediction_write",
                "request_id": validated["request_id"],
                "bucket": bucket,
                "status": "written",
            },
        )
        return object_name
