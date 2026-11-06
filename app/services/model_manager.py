from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.utils.config import ModelConfig

logger = logging.getLogger(__name__)

ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer*.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "*.safetensors",
    "pytorch_model*.bin",
    "*.txt",
    "*.model",
    "vocab.*",
    "*.merges",
    "*.tiktoken",
    "*.bpe",
]
IGNORE_PATTERNS = [
    "tf_model*.h5",
    "*.h5",
    "flax_model*",
    "*.msgpack",
    "*.onnx",
    "onnx/*",
]
_TOKENIZER_RESOURCES = (
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "spm.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)


@dataclass(frozen=True)
class ModelStatus:
    model_key: str
    state: str
    revision: str | None
    local_dir: Path | None
    cache_location: Path | None
    message: str | None = None


class ModelManagerError(RuntimeError):
    """Raised when a model operation cannot be performed safely."""


class ModelManager:
    """Manage revision-qualified Hugging Face snapshots without exposing partial models."""

    def __init__(
        self,
        settings: Any,
        downloader: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._models: tuple[ModelConfig, ...] = tuple(settings.models)
        self._model_by_key = {model.key: model for model in self._models}
        if len(self._model_by_key) != len(self._models):
            raise ModelManagerError("Configured model keys must be unique")
        self._model_root = Path(settings.model_dir).expanduser().resolve()
        self._hf_home = Path(settings.hf_home).expanduser().resolve()
        self._hub_cache = self._hf_home / "hub"
        self._downloader = downloader
        self._states: dict[str, tuple[str, str | None]] = {}
        self._lock = threading.RLock()

    def _model(self, model_key: str) -> ModelConfig:
        try:
            return self._model_by_key[model_key]
        except KeyError as exc:
            raise KeyError(f"Unknown model key: {model_key}") from exc

    @staticmethod
    def _revision_component(revision: str | None) -> str:
        if revision is None:
            return "unpinned"
        if re.fullmatch(r"[A-Fa-f0-9]{40,64}", revision):
            return revision
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip(".-") or "revision"
        suffix = hashlib.sha256(revision.encode("utf-8")).hexdigest()[:10]
        return f"{slug[:48]}-{suffix}"

    def model_path(self, model: ModelConfig | str) -> Path:
        config = self._model(model) if isinstance(model, str) else model
        candidate = (
            self._model_root / config.local_dir / self._revision_component(config.revision)
        ).resolve()
        if not candidate.is_relative_to(self._model_root):
            raise ModelManagerError("Resolved model path is outside MODEL_DIR")
        return candidate

    def cache_path(self, model: ModelConfig | str) -> Path:
        config = self._model(model) if isinstance(model, str) else model
        repository = config.model_id.replace("/", "--")
        return self._hub_cache / f"models--{repository}"

    @staticmethod
    def _required_model_files(directory: Path) -> tuple[str | None, str | None]:
        config_path = directory / "config.json"
        if not config_path.is_file() or config_path.stat().st_size == 0:
            return None, "missing or empty config.json"
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, "invalid config.json model metadata"
        if not isinstance(config_data, dict) or not isinstance(config_data.get("model_type"), str):
            return None, "config.json must contain model_type metadata"

        tokenizer_config = directory / "tokenizer_config.json"
        if not tokenizer_config.is_file() or tokenizer_config.stat().st_size == 0:
            return None, "missing or empty tokenizer_config.json"
        try:
            tokenizer_data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, "invalid tokenizer_config.json metadata"
        if not isinstance(tokenizer_data, dict):
            return None, "tokenizer_config.json must contain an object"

        if not any((directory / resource).is_file() for resource in _TOKENIZER_RESOURCES):
            return None, "missing tokenizer vocabulary or tokenizer.json"
        for resource in _TOKENIZER_RESOURCES:
            resource_path = directory / resource
            if resource_path.is_file() and resource_path.stat().st_size == 0:
                return None, f"empty tokenizer resource: {resource}"

        safetensor_index = directory / "model.safetensors.index.json"
        pytorch_index = directory / "pytorch_model.bin.index.json"
        indexes = [path for path in (safetensor_index, pytorch_index) if path.is_file()]
        if len(indexes) > 1:
            return None, "multiple sharded weight indexes are present"
        if indexes:
            try:
                index_data = json.loads(indexes[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, f"invalid sharded weight index: {indexes[0].name}"
            weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                return None, f"sharded weight index has no weight_map: {indexes[0].name}"
            shard_names = list(weight_map.values())
            if any(not isinstance(name, str) or not name for name in shard_names):
                return None, f"sharded weight index contains invalid shard names: {indexes[0].name}"
            for shard_name in set(shard_names):
                shard = Path(shard_name)
                if shard.is_absolute() or ".." in shard.parts:
                    return None, f"unsafe shard path in index: {indexes[0].name}"
                shard_path = (directory / shard).resolve()
                if not shard_path.is_relative_to(directory.resolve()):
                    return None, f"shard path escapes model directory: {indexes[0].name}"
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    return None, f"missing or empty shard file: {shard_name}"
            return None, None

        weight_patterns = ("*.safetensors", "pytorch_model*.bin")
        weight_files = [
            path
            for pattern in weight_patterns
            for path in directory.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        ]
        if not weight_files:
            return None, "missing or empty PyTorch model weights"
        if any(re.search(r"-\d+-of-\d+", path.name) for path in weight_files):
            return None, "sharded PyTorch weights require a matching weight index"
        return weight_files[0].name, None

    def _validate_snapshot(self, directory: Path) -> tuple[bool, str | None]:
        if not directory.is_dir():
            return False, "model directory is missing"
        _, error = self._required_model_files(directory)
        return error is None, error

    def _status(self, model: ModelConfig) -> ModelStatus:
        local_dir = self.model_path(model)
        cache_location = self.cache_path(model)
        state, message = self._states.get(model.key, ("", None))
        if state == "downloading":
            return ModelStatus(
                model.key, "downloading", model.revision, local_dir, cache_location, message
            )
        if state == "failed":
            return ModelStatus(
                model.key, "failed", model.revision, local_dir, cache_location, message
            )
        ready, validation_error = self._validate_snapshot(local_dir)
        if ready:
            return ModelStatus(model.key, "ready", model.revision, local_dir, cache_location)
        if local_dir.exists():
            return ModelStatus(
                model.key,
                "failed",
                model.revision,
                local_dir,
                cache_location,
                f"Existing model snapshot is incomplete: {validation_error}",
            )
        return ModelStatus(model.key, "not downloaded", model.revision, local_dir, cache_location)

    def status(self, model_key: str) -> ModelStatus:
        with self._lock:
            return self._status(self._model(model_key))

    def list_models(self) -> list[ModelStatus]:
        with self._lock:
            return [self._status(model) for model in self._models]

    def refresh_models(self) -> list[ModelStatus]:
        with self._lock:
            self._states.clear()
            return [self._status(model) for model in self._models]

    def _download_snapshot(self, **kwargs: Any) -> Any:
        if self._downloader is not None:
            return self._downloader(**kwargs)
        os.environ["HF_HOME"] = str(self._hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(
            Path(
                getattr(
                    self._settings,
                    "transformers_cache",
                    self._hf_home / "transformers-cache",
                )
            )
            .expanduser()
            .resolve()
        )
        from huggingface_hub import snapshot_download

        return snapshot_download(**kwargs)

    def download_model(self, model_key: str) -> ModelStatus:
        model = self._model(model_key)
        target = self.model_path(model)
        with self._lock:
            if self._states.get(model.key, ("", None))[0] == "downloading":
                raise ModelManagerError(f"Model {model.key} is already downloading")
            self._states[model.key] = ("downloading", None)
            if target.exists() or target.is_symlink():
                ready, error = self._validate_snapshot(target)
                if ready:
                    self._states.pop(model.key, None)
                    return self._status(model)
                message = f"Existing model snapshot is incomplete: {error}"
                self._states[model.key] = ("failed", message)
                logger.error(
                    "Model reuse rejected",
                    extra={"event": "model_download", "model_key": model.key, "status": "failed"},
                )
                return self._status(model)

        staging: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=target.parent))
            arguments: dict[str, Any] = {
                "repo_id": model.model_id,
                "local_dir": str(staging),
                "cache_dir": str(self._hub_cache),
                "allow_patterns": list(ALLOW_PATTERNS),
                "ignore_patterns": list(IGNORE_PATTERNS),
            }
            if model.revision is not None:
                arguments["revision"] = model.revision
            token = getattr(self._settings, "hf_token", None)
            if token:
                arguments["token"] = token
            self._download_snapshot(**arguments)

            valid, validation_error = self._validate_snapshot(staging)
            if not valid:
                raise ModelManagerError(f"Downloaded snapshot is incomplete: {validation_error}")
            try:
                os.replace(staging, target)
                staging = None
            except OSError:
                if target.exists() and self._validate_snapshot(target)[0]:
                    with self._lock:
                        self._states.pop(model.key, None)
                        status = self._status(model)
                    return status
                raise
            with self._lock:
                self._states.pop(model.key, None)
                status = self._status(model)
            logger.info(
                "Model download completed",
                extra={
                    "event": "model_download",
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "status": "ready",
                },
            )
            return status
        except Exception as exc:
            message = (
                "Model download failed. Check Hugging Face connectivity, model revision, "
                "and access permissions."
            )
            if isinstance(exc, ModelManagerError):
                message = str(exc)
            with self._lock:
                self._states[model.key] = ("failed", message)
                status = self._status(model)
            logger.error(
                "Model download failed",
                extra={
                    "event": "model_download",
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "status": "failed",
                },
            )
            return status
        finally:
            if staging is not None and (staging.exists() or staging.is_symlink()):
                if staging.is_symlink():
                    staging.unlink()
                elif staging.is_dir():
                    shutil.rmtree(staging)

    def delete_model(self, model_key: str) -> ModelStatus:
        model = self._model(model_key)
        target = self.model_path(model)
        with self._lock:
            if self._states.get(model.key, ("", None))[0] == "downloading":
                raise ModelManagerError(f"Cannot delete model {model.key} while it is downloading")
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            self._states.pop(model.key, None)
            logger.info(
                "Managed model deleted",
                extra={"event": "model_delete", "model_key": model.key, "status": "deleted"},
            )
            return self._status(model)

    def clear_huggingface_cache(self) -> str:
        with self._lock:
            try:
                if not self._hub_cache.exists():
                    message = "Hugging Face cache directory is missing; no cache was present."
                else:
                    shutil.rmtree(self._hub_cache)
                    message = "Hugging Face cache cleared."
            except OSError as exc:
                logger.error(
                    "Hugging Face cache cleanup failed",
                    extra={"event": "cache_cleanup", "status": "failed"},
                )
                raise ModelManagerError("Unable to clear the Hugging Face cache") from exc
            logger.info(
                "Hugging Face cache cleanup completed",
                extra={"event": "cache_cleanup", "status": "cleared"},
            )
            return message
