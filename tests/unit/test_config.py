from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.utils.config import ConfigError, load_settings

MODEL_CONFIG = """\
models:
  - key: deberta
    name: DeBERTa
    model_id: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
    revision: 8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8
    local_dir: deberta-v3-base-zeroshot-v2.0
  - key: bart
    name: BART
    model_id: facebook/bart-large-mnli
    revision: d7645e127eaf1aefc7862fd59a17a5aa8558b8ce
    local_dir: bart-large-mnli
"""

LABEL_CONFIG = """\
label_sets:
  - key: sentiment
    name: Sentiment
    labels: [positive, neutral, negative]
default_set: sentiment
"""


def write_configs(
    directory: Path,
    models: str = MODEL_CONFIG,
    labels: str = LABEL_CONFIG,
) -> tuple[Path, Path]:
    models_path = directory / "models.yaml"
    labels_path = directory / "labels.yaml"
    models_path.write_text(models, encoding="utf-8")
    labels_path.write_text(labels, encoding="utf-8")
    return models_path, labels_path


def required_environment() -> dict[str, str]:
    return {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "test-access",
        "MINIO_SECRET_KEY": "test-secret",
    }


def test_loads_required_environment_and_documented_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_path, labels_path = write_configs(tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    settings = load_settings(
        required_environment(),
        models_path=models_path,
        labels_path=labels_path,
        load_dotenv_file=False,
    )

    assert settings.minio_bucket == "zeroshot-learning-output"
    assert settings.model_dir == Path("models/downloaded")
    assert settings.hf_home == Path("models/hf-home")
    assert settings.transformers_cache == Path("models/transformers-cache")
    assert settings.app_version == "0.1.0"
    assert len(settings.models) == 2
    assert settings.label_sets[0].key == "sentiment"
    assert settings.default_set == "sentiment"
    assert Path(os.environ["HF_HOME"]) == Path("models/hf-home")
    assert Path(os.environ["TRANSFORMERS_CACHE"]) == Path("models/transformers-cache")


@pytest.mark.parametrize("missing", ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"])
def test_missing_required_minio_environment_is_rejected(tmp_path: Path, missing: str) -> None:
    models_path, labels_path = write_configs(tmp_path)
    environment = required_environment()
    environment.pop(missing)

    with pytest.raises(ConfigError, match=missing):
        load_settings(
            environment,
            models_path=models_path,
            labels_path=labels_path,
            load_dotenv_file=False,
        )


def test_secret_values_are_not_revealed_by_settings_repr(tmp_path: Path) -> None:
    models_path, labels_path = write_configs(tmp_path)
    settings = load_settings(
        required_environment(),
        models_path=models_path,
        labels_path=labels_path,
        load_dotenv_file=False,
    )

    assert "test-access" not in repr(settings)
    assert "test-secret" not in repr(settings)


def test_rejects_duplicate_model_keys_and_unsafe_local_dirs(tmp_path: Path) -> None:
    duplicate_keys = MODEL_CONFIG.replace("key: bart", "key: deberta")
    models_path, labels_path = write_configs(tmp_path, models=duplicate_keys)
    with pytest.raises(ConfigError, match="duplicate model key"):
        load_settings(
            required_environment(),
            models_path=models_path,
            labels_path=labels_path,
            load_dotenv_file=False,
        )

    unsafe_paths = MODEL_CONFIG.replace("local_dir: bart-large-mnli", "local_dir: ../bart")
    models_path, labels_path = write_configs(tmp_path, models=unsafe_paths)
    with pytest.raises(ConfigError, match="local_dir"):
        load_settings(
            required_environment(),
            models_path=models_path,
            labels_path=labels_path,
            load_dotenv_file=False,
        )


def test_rejects_unverified_baseline_revision_and_invalid_labels(tmp_path: Path) -> None:
    models_path, labels_path = write_configs(
        tmp_path, models=MODEL_CONFIG.replace("8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8", "main")
    )
    with pytest.raises(ConfigError, match="baseline revision"):
        load_settings(
            required_environment(),
            models_path=models_path,
            labels_path=labels_path,
            load_dotenv_file=False,
        )

    invalid_labels = LABEL_CONFIG.replace("[positive, neutral, negative]", "[positive, POSITIVE]")
    models_path, labels_path = write_configs(tmp_path, labels=invalid_labels)
    with pytest.raises(ConfigError, match="duplicate"):
        load_settings(
            required_environment(),
            models_path=models_path,
            labels_path=labels_path,
            load_dotenv_file=False,
        )


def test_rejects_malformed_yaml_without_replacing_it(tmp_path: Path) -> None:
    malformed = "models: [\n"
    models_path, labels_path = write_configs(tmp_path, models=malformed)

    with pytest.raises(ConfigError, match="models configuration"):
        load_settings(
            required_environment(),
            models_path=models_path,
            labels_path=labels_path,
            load_dotenv_file=False,
        )

    assert models_path.read_text(encoding="utf-8") == malformed
