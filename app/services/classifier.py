from __future__ import annotations

import gc
import importlib
import logging
import math
import os
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.minio_storage import create_request_id
from app.utils.config import ModelConfig

logger = logging.getLogger(__name__)
MAX_INPUT_LENGTH = 10_000
_ALLOWED_CONTROLS = {"\t", "\n", "\r"}


class ClassificationError(RuntimeError):
    """Raised when a classification request cannot be completed."""


class Classifier:
    """Run zero-shot inference with one cached PyTorch pipeline at a time."""

    def __init__(
        self,
        settings: Any,
        model_manager: Any,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._model_manager = model_manager
        self._models: dict[str, ModelConfig] = {model.key: model for model in settings.models}
        self._pipeline_factory = pipeline_factory
        self._active_model_key: str | None = None
        self._active_pipeline: Any | None = None
        self._lock = threading.RLock()

    @property
    def active_model_key(self) -> str | None:
        return self._active_model_key

    @staticmethod
    def _validate_text(input_text: str) -> str:
        if not isinstance(input_text, str):
            raise ClassificationError("Input text must be plain text.")
        cleaned = input_text.strip()
        if not cleaned:
            raise ClassificationError("Input text cannot be empty.")
        if len(cleaned) > MAX_INPUT_LENGTH:
            raise ClassificationError("Input text cannot exceed 10,000 characters.")
        if any(
            unicodedata.category(character) == "Cc" and character not in _ALLOWED_CONTROLS
            for character in cleaned
        ):
            raise ClassificationError("Input text contains disallowed control characters.")
        return cleaned

    @staticmethod
    def _validate_labels(candidate_labels: Sequence[str]) -> list[str]:
        if isinstance(candidate_labels, str) or not isinstance(candidate_labels, Sequence):
            raise ClassificationError(
                "Select a saved candidate-label set containing at least two labels."
            )
        if len(candidate_labels) < 2:
            raise ClassificationError("At least two candidate labels are required.")
        labels: list[str] = []
        seen: set[str] = set()
        for label in candidate_labels:
            if not isinstance(label, str):
                raise ClassificationError("Candidate labels must be plain text.")
            cleaned = label.strip()
            if not cleaned:
                raise ClassificationError("Candidate labels cannot be empty.")
            folded = cleaned.casefold()
            if folded in seen:
                raise ClassificationError("Candidate labels must be unique.")
            labels.append(cleaned)
            seen.add(folded)
        return labels

    def _get_model(self, model_key: str) -> ModelConfig:
        model = self._models.get(model_key)
        if model is None:
            raise ClassificationError("Select a configured model.")
        try:
            status = self._model_manager.status(model_key)
        except (KeyError, OSError) as exc:
            raise ClassificationError("The selected model is unavailable.") from exc
        if status.state != "ready":
            raise ClassificationError(
                f"The selected model is not downloaded or ready (status: {status.state})."
            )
        if status.local_dir is None:
            raise ClassificationError("The selected model has no available local directory.")
        return model

    def _create_pipeline(self, model: ModelConfig, local_dir: Path) -> Any:
        if self._pipeline_factory is None:
            os.environ["HF_HOME"] = str(Path(self._settings.hf_home).expanduser().resolve())
            os.environ["TRANSFORMERS_CACHE"] = str(
                Path(self._settings.transformers_cache).expanduser().resolve()
            )
            try:
                importlib.import_module("torch")
                from transformers import pipeline
            except (ImportError, ModuleNotFoundError) as exc:
                raise ClassificationError(
                    "PyTorch is unavailable; install the required PyTorch backend before inference."
                ) from exc
            factory = pipeline
        else:
            factory = self._pipeline_factory

        try:
            return factory(
                task="zero-shot-classification",
                model=str(local_dir),
                tokenizer=str(local_dir),
                framework="pt",
                device=-1,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ClassificationError(
                "PyTorch is unavailable; inference requires the PyTorch backend."
            ) from exc
        except Exception as exc:
            raise ClassificationError(
                f"Unable to load the selected model with the PyTorch backend: {model.name}."
            ) from exc

    def _pipeline_for(self, model: ModelConfig, local_dir: Path) -> Any:
        if self._active_model_key == model.key and self._active_pipeline is not None:
            return self._active_pipeline

        self._active_pipeline = None
        self._active_model_key = None
        gc.collect()
        pipeline = self._create_pipeline(model, local_dir)
        self._active_pipeline = pipeline
        self._active_model_key = model.key
        return pipeline

    @staticmethod
    def _rank_results(result: object, candidate_labels: list[str]) -> list[dict[str, str | float]]:
        if not isinstance(result, dict):
            raise ClassificationError("The model returned an invalid classification result.")
        labels = result.get("labels")
        scores = result.get("scores")
        if not isinstance(labels, list) or not isinstance(scores, list):
            raise ClassificationError("The model returned an invalid label or score list.")
        if len(labels) != len(candidate_labels) or len(scores) != len(candidate_labels):
            raise ClassificationError("The model did not return a score for every candidate label.")
        if any(not isinstance(label, str) for label in labels):
            raise ClassificationError("The model returned an invalid candidate label.")
        if len(set(labels)) != len(labels) or set(labels) != set(candidate_labels):
            raise ClassificationError(
                "The model returned labels that do not match the selected set."
            )

        scores_by_label: dict[str, float] = {}
        for label, score in zip(labels, scores, strict=True):
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise ClassificationError("The model returned a non-numeric classification score.")
            normalized = float(score)
            if not math.isfinite(normalized) or not 0 <= normalized <= 1:
                raise ClassificationError("The model returned a score outside the range [0, 1].")
            scores_by_label[label] = normalized
        return [
            {"label": label, "score": scores_by_label[label]}
            for label in sorted(
                candidate_labels, key=lambda value: scores_by_label[value], reverse=True
            )
        ]

    def classify(
        self,
        input_text: str,
        model_key: str,
        candidate_labels: Sequence[str],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        cleaned_text = self._validate_text(input_text)
        labels = self._validate_labels(candidate_labels)
        model = self._get_model(model_key)
        status = self._model_manager.status(model_key)

        with self._lock:
            classifier = self._pipeline_for(model, Path(status.local_dir))
            started = time.perf_counter()
            try:
                result = classifier(cleaned_text, candidate_labels=labels)
            except Exception as exc:
                raise ClassificationError(
                    f"Inference failed for the selected model: {model.name}."
                ) from exc
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            ranked = self._rank_results(result, labels)

        now = datetime.now(UTC)
        record: dict[str, Any] = {
            "request_id": create_request_id(),
            "timestamp": now.isoformat(),
            "input_text": cleaned_text,
            "predicted_class": ranked[0]["label"],
            "confidence": ranked[0]["score"],
            "candidate_labels": labels,
            "ranked_predictions": ranked,
            "model_name": model.name,
            "model_id": model.model_id,
            "application_version": self._settings.app_version,
            "latency_ms": latency_ms,
        }
        if model.revision is not None:
            record["model_revision"] = model.revision
        if session_id:
            record["session_id"] = session_id
        logger.info(
            "Classification inference completed",
            extra={
                "event": "inference",
                "request_id": record["request_id"],
                "model_id": model.model_id,
                "model_key": model.key,
                "status": "inferred",
            },
        )
        return record
