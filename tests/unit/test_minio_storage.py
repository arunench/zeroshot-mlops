from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.minio_storage import MinIOStorage, StorageError, create_request_id


class FakeMinio:
    def __init__(self, *, exists: bool = False, fail_write: bool = False) -> None:
        self.exists = exists
        self.fail_write = fail_write
        self.created: list[str] = []
        self.objects: list[dict[str, Any]] = []

    def bucket_exists(self, bucket: str) -> bool:
        return self.exists

    def make_bucket(self, bucket: str) -> None:
        self.created.append(bucket)
        self.exists = True

    def put_object(
        self, bucket: str, object_name: str, data: BytesIO, length: int, content_type: str
    ) -> None:
        if self.fail_write:
            raise RuntimeError("simulated MinIO failure")
        payload = data.read()
        assert len(payload) == length
        self.objects.append(
            {
                "bucket": bucket,
                "object_name": object_name,
                "payload": payload,
                "content_type": content_type,
            }
        )


def storage(client: FakeMinio) -> MinIOStorage:
    settings = SimpleNamespace(
        minio_endpoint="localhost:9000",
        minio_access_key="test-access",
        minio_secret_key="test-secret",
        minio_secure=False,
        minio_bucket="zeroshot-learning-output",
    )
    return MinIOStorage(settings, client=client)


def prediction(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "timestamp": "2026-11-05T23:10:00Z",
        "input_text": "The release is excellent.",
        "predicted_class": "positive",
        "confidence": 0.9,
        "candidate_labels": ["positive", "negative"],
        "ranked_predictions": [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.1},
        ],
        "model_name": "DeBERTa",
        "model_id": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "model_revision": "verified-revision",
        "application_version": "0.1.0",
    }


def test_creates_missing_bucket_and_writes_utc_partitioned_json() -> None:
    client = FakeMinio()
    result = storage(client).write_prediction(prediction("request-1"))

    assert client.created == ["zeroshot-learning-output"]
    assert result == "predictions/2026/11/05/request-1.json"
    assert len(client.objects) == 1
    stored = client.objects[0]
    assert stored["bucket"] == "zeroshot-learning-output"
    assert stored["content_type"] == "application/json"
    assert json.loads(stored["payload"])["model_revision"] == "verified-revision"


def test_reuses_existing_bucket_and_omits_unset_revision() -> None:
    client = FakeMinio(exists=True)
    record = prediction("request-2")
    record.pop("model_revision")

    storage(client).write_prediction(record)

    assert client.created == []
    assert "model_revision" not in json.loads(client.objects[0]["payload"])


def test_generated_request_ids_are_unique() -> None:
    first = create_request_id()
    second = create_request_id()

    assert first != second


def test_rejects_invalid_timestamp_and_out_of_range_scores() -> None:
    client = FakeMinio(exists=True)
    record = prediction("request-3")
    record["timestamp"] = datetime.now().replace(tzinfo=None).isoformat()
    with pytest.raises(StorageError, match="UTC"):
        storage(client).write_prediction(record)

    record = prediction("request-4")
    record["confidence"] = 1.1
    with pytest.raises(StorageError, match="confidence"):
        storage(client).write_prediction(record)


def test_minio_write_failure_is_explicit_and_not_reported_as_success() -> None:
    service = storage(FakeMinio(exists=True, fail_write=True))

    with pytest.raises(StorageError, match="MinIO"):
        service.write_prediction(prediction("request-5"))
