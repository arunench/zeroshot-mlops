from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_MINIO_BUCKET = "zeroshot-learning-output"
REQUIRED_BASELINES = {
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
    "facebook/bart-large-mnli": "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
}


class ConfigError(ValueError):
    """Raised when environment or project configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    model_id: str
    revision: str | None
    local_dir: str


@dataclass(frozen=True)
class LabelSetConfig:
    key: str
    name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    minio_endpoint: str
    minio_access_key: str = field(repr=False)
    minio_secret_key: str = field(repr=False)
    minio_secure: bool = False
    minio_bucket: str = DEFAULT_MINIO_BUCKET
    model_dir: Path = Path("models/downloaded")
    hf_home: Path = Path("models/hf-home")
    transformers_cache: Path = Path("models/transformers-cache")
    hf_token: str | None = field(default=None, repr=False)
    app_version: str = "0.1.0"
    models: tuple[ModelConfig, ...] = ()
    label_sets: tuple[LabelSetConfig, ...] = ()
    default_set: str | None = None


def _read_yaml(path: Path, label: str) -> object:
    try:
        with path.open(encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to read {label} configuration at {path}: {exc}") from exc


def _required(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key, "").strip()
    if not value:
        raise ConfigError(f"Required environment variable {key} is missing or empty")
    return value


def _safe_key(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigError(f"{field_name} must be a non-empty filesystem-safe identifier")
    return value


def _parse_models(data: object) -> tuple[ModelConfig, ...]:
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ConfigError("models configuration must contain a 'models' list")

    parsed: list[ModelConfig] = []
    seen_keys: set[str] = set()
    seen_dirs: set[str] = set()
    seen_ids: set[str] = set()
    for item in data["models"]:
        if not isinstance(item, dict):
            raise ConfigError("Each model configuration must be a mapping")
        key = _safe_key(item.get("key"), "Model key")
        name = item.get("name")
        model_id = item.get("model_id")
        local_dir = item.get("local_dir")
        revision = item.get("revision")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"Model {key} requires a non-empty name")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigError(f"Model {key} requires a Hugging Face model_id")
        if not isinstance(local_dir, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", local_dir
        ):
            raise ConfigError(f"Model {key} local_dir must be a safe single path component")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ConfigError(f"Model {key} revision must be a non-empty string when provided")
        if key in seen_keys:
            raise ConfigError(f"Configuration contains duplicate model key: {key}")
        if local_dir.casefold() in seen_dirs:
            raise ConfigError(f"Configuration contains duplicate local_dir: {local_dir}")
        if model_id in seen_ids:
            raise ConfigError(f"Configuration contains duplicate model_id: {model_id}")
        expected_revision = REQUIRED_BASELINES.get(model_id)
        if expected_revision and revision != expected_revision:
            raise ConfigError(
                f"Model {model_id} must use its verified baseline revision {expected_revision}"
            )
        seen_keys.add(key)
        seen_dirs.add(local_dir.casefold())
        seen_ids.add(model_id)
        parsed.append(
            ModelConfig(
                key=key,
                name=name.strip(),
                model_id=model_id.strip(),
                revision=revision,
                local_dir=local_dir,
            )
        )

    missing_baselines = set(REQUIRED_BASELINES).difference(seen_ids)
    if missing_baselines:
        raise ConfigError(
            "models configuration is missing required baseline model(s): "
            + ", ".join(sorted(missing_baselines))
        )
    return tuple(parsed)


def _parse_labels(data: object) -> tuple[tuple[LabelSetConfig, ...], str | None]:
    if not isinstance(data, dict) or not isinstance(data.get("label_sets"), list):
        raise ConfigError("labels configuration must contain a 'label_sets' list")

    parsed: list[LabelSetConfig] = []
    seen_keys: set[str] = set()
    for item in data["label_sets"]:
        if not isinstance(item, dict):
            raise ConfigError("Each label set configuration must be a mapping")
        key = _safe_key(item.get("key"), "Label-set key")
        name = item.get("name")
        labels = item.get("labels")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"Label set {key} requires a non-empty name")
        if not isinstance(labels, list):
            raise ConfigError(f"Label set {key} labels must be a list")
        normalized: list[str] = []
        seen_labels: set[str] = set()
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ConfigError(f"Label set {key} cannot contain empty labels")
            trimmed = label.strip()
            folded = trimmed.casefold()
            if folded in seen_labels:
                raise ConfigError(f"Label set {key} contains a duplicate label: {trimmed}")
            seen_labels.add(folded)
            normalized.append(trimmed)
        if len(normalized) < 2:
            raise ConfigError(f"Label set {key} must contain at least two labels")
        if key in seen_keys:
            raise ConfigError(f"Configuration contains duplicate label-set key: {key}")
        seen_keys.add(key)
        parsed.append(LabelSetConfig(key=key, name=name.strip(), labels=tuple(normalized)))

    default_set = data.get("default_set")
    if default_set is not None:
        if not isinstance(default_set, str) or default_set not in seen_keys:
            raise ConfigError("default_set must reference an existing label-set key")
    return tuple(parsed), default_set


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    models_path: Path | str = Path("configs/models.yaml"),
    labels_path: Path | str = Path("configs/labels.yaml"),
    load_dotenv_file: bool = True,
) -> Settings:
    """Load validated settings and export Hugging Face cache paths before model imports."""
    if environ is None:
        if load_dotenv_file:
            load_dotenv(override=False)
        values: Mapping[str, str] = os.environ
    else:
        values = environ

    model_dir = Path(values.get("MODEL_DIR", "models/downloaded"))
    hf_home = Path(values.get("HF_HOME", "models/hf-home"))
    transformers_cache = Path(values.get("TRANSFORMERS_CACHE", "models/transformers-cache"))
    if environ is None:
        os.environ["MODEL_DIR"] = str(model_dir)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    else:
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    secure_value = values.get("MINIO_SECURE", "false").strip().casefold()
    if secure_value not in {"true", "false"}:
        raise ConfigError("MINIO_SECURE must be 'true' or 'false'")

    models = _parse_models(_read_yaml(Path(models_path), "models"))
    label_sets, default_set = _parse_labels(_read_yaml(Path(labels_path), "labels"))

    return Settings(
        minio_endpoint=_required(values, "MINIO_ENDPOINT"),
        minio_access_key=_required(values, "MINIO_ACCESS_KEY"),
        minio_secret_key=_required(values, "MINIO_SECRET_KEY"),
        minio_secure=secure_value == "true",
        minio_bucket=values.get("MINIO_BUCKET", DEFAULT_MINIO_BUCKET).strip()
        or DEFAULT_MINIO_BUCKET,
        model_dir=model_dir,
        hf_home=hf_home,
        transformers_cache=transformers_cache,
        hf_token=values.get("HF_TOKEN") or None,
        app_version=values.get("APP_VERSION", "0.1.0"),
        models=models,
        label_sets=label_sets,
        default_set=default_set,
    )
