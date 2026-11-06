from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.services.classifier import ClassificationError, Classifier
from app.services.model_manager import ModelStatus
from app.utils.config import ModelConfig

MODEL_A = ModelConfig(
    key="model-a",
    name="Model A",
    model_id="org/model-a",
    revision="a" * 40,
    local_dir="model-a",
)
MODEL_B = ModelConfig(
    key="model-b",
    name="Model B",
    model_id="org/model-b",
    revision="b" * 40,
    local_dir="model-b",
)


@dataclass(frozen=True)
class Settings:
    models: tuple[ModelConfig, ...]
    model_dir: Path
    app_version: str = "1.2.3"


class ModelManagerStub:
    def __init__(
        self,
        models: tuple[ModelConfig, ...],
        model_dir: Path,
        ready: set[str] | None = None,
    ):
        self.models = models
        self.model_dir = model_dir
        self.ready = ready if ready is not None else {model.key for model in models}

    def status(self, model_key: str) -> ModelStatus:
        model = next((model for model in self.models if model.key == model_key), None)
        if model is None:
            raise KeyError(f"Unknown model key: {model_key}")
        model_dir = self.model_dir / model.local_dir / (model.revision or "unpinned")
        return ModelStatus(
            model_key=model.key,
            state="ready" if model_key in self.ready else "not downloaded",
            revision=model.revision,
            local_dir=model_dir,
            cache_location=Path("cache"),
        )


class PipelineStub:
    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.output = output or {"labels": ["positive", "negative"], "scores": [0.8, 0.2]}

    def __call__(self, text: str, candidate_labels: list[str]) -> dict[str, Any]:
        self.calls.append({"text": text, "candidate_labels": candidate_labels})
        return self.output


def build_classifier(
    tmp_path: Path,
    pipeline_factory: Any | None = None,
    *,
    models: tuple[ModelConfig, ...] = (MODEL_A, MODEL_B),
    ready: set[str] | None = None,
) -> tuple[Classifier, ModelManagerStub]:
    model_root = tmp_path / "models"
    model_manager = ModelManagerStub(models, model_root, ready)
    classifier = Classifier(
        Settings(models=models, model_dir=model_root),
        model_manager,
        pipeline_factory=pipeline_factory,
    )
    return classifier, model_manager


def test_trims_input_and_returns_complete_prediction_metadata(tmp_path: Path) -> None:
    pipeline = PipelineStub(
        {"labels": ["positive", "neutral", "negative"], "scores": [0.7, 0.2, 0.1]}
    )
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    record = classifier.classify(
        "  This is good.  ",
        model_key="model-a",
        candidate_labels=["positive", "neutral", "negative"],
        session_id="session-1",
    )

    assert record["input_text"] == "This is good."
    assert record["predicted_class"] == "positive"
    assert record["confidence"] == 0.7
    assert record["candidate_labels"] == ["positive", "neutral", "negative"]
    assert record["ranked_predictions"] == [
        {"label": "positive", "score": 0.7},
        {"label": "neutral", "score": 0.2},
        {"label": "negative", "score": 0.1},
    ]
    assert record["request_id"]
    assert datetime.fromisoformat(record["timestamp"]).utcoffset() == UTC.utcoffset(None)
    assert record["model_name"] == MODEL_A.name
    assert record["model_id"] == MODEL_A.model_id
    assert record["model_revision"] == MODEL_A.revision
    assert record["application_version"] == "1.2.3"
    assert record["session_id"] == "session-1"
    assert record["latency_ms"] >= 0
    assert pipeline.calls[0]["text"] == "This is good."


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_rejects_empty_or_whitespace_text_before_inference(tmp_path: Path, text: str) -> None:
    pipeline = PipelineStub()
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    with pytest.raises(ClassificationError, match="empty"):
        classifier.classify(text, "model-a", ["yes", "no"])

    assert pipeline.calls == []


def test_rejects_text_over_10000_characters_before_inference(tmp_path: Path) -> None:
    pipeline = PipelineStub()
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    with pytest.raises(ClassificationError, match="10,000"):
        classifier.classify("x" * 10_001, "model-a", ["yes", "no"])

    assert pipeline.calls == []


@pytest.mark.parametrize("text", ["bad\x00input", "bad\x07input", "bad\u0085input"])
def test_rejects_disallowed_control_characters_before_inference(tmp_path: Path, text: str) -> None:
    pipeline = PipelineStub()
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    with pytest.raises(ClassificationError, match="control"):
        classifier.classify(text, "model-a", ["yes", "no"])

    assert pipeline.calls == []


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["only one"],
        ["yes", ""],
        ["yes", " YES "],
    ],
)
def test_rejects_invalid_candidate_labels_before_inference(
    tmp_path: Path, labels: list[str]
) -> None:
    pipeline = PipelineStub()
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    with pytest.raises(ClassificationError):
        classifier.classify("valid input", "model-a", labels)

    assert pipeline.calls == []


def test_selects_pytorch_backend_and_uses_local_revision_directory(tmp_path: Path) -> None:
    pipeline = PipelineStub()
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> PipelineStub:
        factory_calls.append(kwargs)
        return pipeline

    classifier, _ = build_classifier(tmp_path, factory, models=(MODEL_A,))
    classifier.classify("valid", "model-a", ["positive", "negative"])

    assert factory_calls == [
        {
            "task": "zero-shot-classification",
            "model": str((tmp_path / "models" / MODEL_A.local_dir / MODEL_A.revision).resolve()),
            "tokenizer": str(
                (tmp_path / "models" / MODEL_A.local_dir / MODEL_A.revision).resolve()
            ),
            "framework": "pt",
            "device": -1,
        }
    ]


def test_pytorch_import_failure_is_clear_and_does_not_retry_other_backend(
    tmp_path: Path,
) -> None:
    factory_calls: list[dict[str, Any]] = []

    def unavailable_pytorch(**kwargs: Any) -> Any:
        factory_calls.append(kwargs)
        raise ImportError("torch is not installed")

    classifier, _ = build_classifier(tmp_path, unavailable_pytorch, models=(MODEL_A,))

    with pytest.raises(ClassificationError, match="PyTorch"):
        classifier.classify("valid", "model-a", ["positive", "negative"])

    assert len(factory_calls) == 1
    assert factory_calls[0]["framework"] == "pt"


def test_reuses_pipeline_for_consecutive_predictions_of_same_model(tmp_path: Path) -> None:
    pipeline = PipelineStub()
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> PipelineStub:
        factory_calls.append(kwargs)
        return pipeline

    classifier, _ = build_classifier(tmp_path, factory, models=(MODEL_A,))
    classifier.classify("first", "model-a", ["positive", "negative"])
    classifier.classify("second", "model-a", ["positive", "negative"])

    assert len(factory_calls) == 1
    assert len(pipeline.calls) == 2


def test_switches_models_and_keeps_only_the_latest_pipeline(tmp_path: Path) -> None:
    pipelines = {"model-a": PipelineStub(), "model-b": PipelineStub()}
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> PipelineStub:
        factory_calls.append(kwargs)
        return pipelines["model-a" if "model-a" in kwargs["model"] else "model-b"]

    classifier, _ = build_classifier(tmp_path, factory)
    classifier.classify("first", "model-a", ["positive", "negative"])
    classifier.classify("second", "model-b", ["positive", "negative"])
    classifier.classify("third", "model-b", ["positive", "negative"])

    assert len(factory_calls) == 2
    assert classifier.active_model_key == "model-b"
    assert len(pipelines["model-a"].calls) == 1
    assert len(pipelines["model-b"].calls) == 2


def test_requires_ready_configured_model(tmp_path: Path) -> None:
    classifier, _ = build_classifier(
        tmp_path, lambda **_: PipelineStub(), models=(MODEL_A,), ready=set()
    )

    with pytest.raises(ClassificationError, match="not downloaded"):
        classifier.classify("valid", "model-a", ["yes", "no"])
    with pytest.raises(ClassificationError, match="configured"):
        classifier.classify("valid", "missing", ["yes", "no"])


@pytest.mark.parametrize(
    "scores",
    [
        [1.1, -0.1],
        [math.nan, 0.5],
        [True, 0.5],
    ],
)
def test_rejects_scores_outside_valid_probability_range(
    tmp_path: Path, scores: list[float]
) -> None:
    pipeline = PipelineStub({"labels": ["yes", "no"], "scores": scores})
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    with pytest.raises(ClassificationError, match="score"):
        classifier.classify("valid", "model-a", ["yes", "no"])


def test_ranks_all_labels_by_descending_score(tmp_path: Path) -> None:
    pipeline = PipelineStub({"labels": ["low", "high", "middle"], "scores": [0.1, 0.7, 0.2]})
    classifier, _ = build_classifier(tmp_path, lambda **_: pipeline, models=(MODEL_A,))

    record = classifier.classify("valid", "model-a", ["high", "middle", "low"])

    assert record["ranked_predictions"] == [
        {"label": "high", "score": 0.7},
        {"label": "middle", "score": 0.2},
        {"label": "low", "score": 0.1},
    ]
    assert [row["label"] for row in record["ranked_predictions"]] == record["candidate_labels"]
    assert record["confidence"] == record["ranked_predictions"][0]["score"]


@pytest.mark.parametrize(
    "result",
    [
        {"labels": ["yes"], "scores": [0.9]},
        {"labels": ["unknown", "yes"], "scores": [0.9, 0.1]},
        {"labels": ["yes", "yes"], "scores": [0.9, 0.1]},
        {"labels": ["yes", "no"], "scores": [0.9]},
        {"labels": ["yes", "no"], "scores": ["high", 0.1]},
    ],
)
def test_rejects_malformed_or_incomplete_pipeline_results(
    tmp_path: Path, result: dict[str, Any]
) -> None:
    classifier, _ = build_classifier(tmp_path, lambda **_: PipelineStub(result), models=(MODEL_A,))

    with pytest.raises(ClassificationError):
        classifier.classify("valid", "model-a", ["yes", "no"])


def test_omits_optional_revision_and_session_when_unset(tmp_path: Path) -> None:
    unpinned = ModelConfig(
        key="extra",
        name="Extra",
        model_id="org/extra",
        revision=None,
        local_dir="extra",
    )
    classifier, _ = build_classifier(
        tmp_path,
        lambda **_: PipelineStub({"labels": ["yes", "no"], "scores": [0.8, 0.2]}),
        models=(unpinned,),
    )

    record = classifier.classify("valid", "extra", ["yes", "no"])

    assert "model_revision" not in record
    assert "session_id" not in record
    assert datetime.fromisoformat(record["timestamp"]).tzinfo == UTC
