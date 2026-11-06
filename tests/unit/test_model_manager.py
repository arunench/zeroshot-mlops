from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from app.services.model_manager import ModelManager, ModelStatus
from app.utils.config import ModelConfig

DEBERTA = ModelConfig(
    key="deberta",
    name="DeBERTa",
    model_id="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    revision="8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
    local_dir="deberta-v3-base-zeroshot-v2.0",
)
BART = ModelConfig(
    key="bart",
    name="BART",
    model_id="facebook/bart-large-mnli",
    revision="d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
    local_dir="bart-large-mnli",
)


@dataclass(frozen=True)
class ManagerSettings:
    models: tuple[ModelConfig, ...]
    model_dir: Path
    hf_home: Path
    transformers_cache: Path = Path("models/transformers-cache")


def build_valid_snapshot(directory: Path, model: ModelConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "deberta" if model == DEBERTA else "bart"}),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "FixtureTokenizer"}), encoding="utf-8"
    )
    (directory / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"fixture weights")


def make_manager(
    tmp_path: Path,
    downloader: Any,
    models: tuple[ModelConfig, ...] = (DEBERTA, BART),
) -> ModelManager:
    return ModelManager(
        ManagerSettings(
            models=models,
            model_dir=tmp_path / "managed-models",
            hf_home=tmp_path / "hf-home",
            transformers_cache=tmp_path / "hf-home" / "transformers-cache",
        ),
        downloader=downloader,
    )


def test_list_configured_models_and_revision_paths(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, downloader=lambda **_: pytest.fail("unexpected download"))

    statuses = manager.list_models()

    assert [status.model_key for status in statuses] == ["deberta", "bart"]
    assert all(status.state == "not downloaded" for status in statuses)
    assert statuses[0].revision == DEBERTA.revision
    assert statuses[0].local_dir == (
        tmp_path / "managed-models" / DEBERTA.local_dir / DEBERTA.revision
    )
    assert statuses[0].cache_location == (
        tmp_path / "hf-home" / "hub" / "models--MoritzLaurer--deberta-v3-base-zeroshot-v2.0"
    )


def test_download_uses_revision_and_reuses_complete_local_model(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def downloader(**kwargs: Any) -> None:
        calls.append(kwargs)
        build_valid_snapshot(Path(kwargs["local_dir"]), DEBERTA)

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))

    first = manager.download_model("deberta")
    second = manager.download_model("deberta")

    assert first.state == "ready"
    assert second.state == "ready"
    assert len(calls) == 1
    assert calls[0]["repo_id"] == DEBERTA.model_id
    assert calls[0]["revision"] == DEBERTA.revision
    assert calls[0]["cache_dir"] == str(tmp_path / "hf-home" / "hub")
    assert "model.safetensors.index.json" in calls[0]["allow_patterns"]
    assert "pytorch_model.bin.index.json" in calls[0]["allow_patterns"]
    assert calls[0]["ignore_patterns"]
    assert not list((tmp_path / "managed-models").rglob(".staging-*"))


def test_download_stages_then_promotes_complete_snapshot(tmp_path: Path) -> None:
    staging_paths: list[Path] = []

    def downloader(**kwargs: Any) -> None:
        staging = Path(kwargs["local_dir"])
        staging_paths.append(staging)
        assert staging.name.startswith(".staging-")
        build_valid_snapshot(staging, BART)

    manager = make_manager(tmp_path, downloader, models=(BART,))

    result = manager.download_model("bart")

    assert result.state == "ready"
    assert staging_paths[0] != result.local_dir
    assert result.local_dir is not None and (result.local_dir / "model.safetensors").is_file()
    assert not staging_paths[0].exists()


def test_status_is_downloading_while_snapshot_is_being_fetched(tmp_path: Path) -> None:
    manager: ModelManager
    observed_states: list[str] = []

    def downloader(**kwargs: Any) -> None:
        observed_states.append(manager.status("deberta").state)
        build_valid_snapshot(Path(kwargs["local_dir"]), DEBERTA)

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))

    result = manager.download_model("deberta")

    assert observed_states == ["downloading"]
    assert result.state == "ready"


@pytest.mark.parametrize("broken_shard", ["missing", "empty"])
def test_sharded_index_rejects_missing_or_empty_shards(tmp_path: Path, broken_shard: str) -> None:
    def downloader(**kwargs: Any) -> None:
        staging = Path(kwargs["local_dir"])
        build_valid_snapshot(staging, DEBERTA)
        (staging / "model.safetensors").unlink()
        (staging / "model-00001-of-00002.safetensors").write_bytes(b"shard one")
        if broken_shard == "empty":
            (staging / "model-00002-of-00002.safetensors").touch()
        index = {
            "metadata": {"total_size": 20},
            "weight_map": {
                "embeddings.weight": "model-00001-of-00002.safetensors",
                "encoder.weight": "model-00002-of-00002.safetensors",
            },
        }
        (staging / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))

    result = manager.download_model("deberta")

    assert result.state == "failed"
    assert "shard" in (result.message or "").casefold()
    assert result.local_dir is not None and not result.local_dir.exists()


def test_sharded_index_accepts_all_nonempty_shards(tmp_path: Path) -> None:
    def downloader(**kwargs: Any) -> None:
        staging = Path(kwargs["local_dir"])
        build_valid_snapshot(staging, DEBERTA)
        (staging / "model.safetensors").unlink()
        (staging / "model-00001-of-00002.safetensors").write_bytes(b"one")
        (staging / "model-00002-of-00002.safetensors").write_bytes(b"two")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layer.one": "model-00001-of-00002.safetensors",
                        "layer.two": "model-00002-of-00002.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))

    result = manager.download_model("deberta")

    assert result.state == "ready"
    assert result.local_dir is not None
    assert (result.local_dir / "model-00001-of-00002.safetensors").stat().st_size > 0


def test_sharded_weights_without_index_are_not_ready(tmp_path: Path) -> None:
    def downloader(**kwargs: Any) -> None:
        staging = Path(kwargs["local_dir"])
        build_valid_snapshot(staging, DEBERTA)
        (staging / "model.safetensors").unlink()
        (staging / "model-00001-of-00002.safetensors").write_bytes(b"shard one")

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))

    result = manager.download_model("deberta")

    assert result.state == "failed"
    assert "index" in (result.message or "").casefold()


@pytest.mark.parametrize("broken_shard", ["missing", "empty"])
def test_incomplete_sharded_local_snapshot_is_not_reused(tmp_path: Path, broken_shard: str) -> None:
    calls: list[dict[str, Any]] = []

    def downloader(**kwargs: Any) -> None:
        calls.append(kwargs)

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))
    target = manager.model_path(DEBERTA)
    build_valid_snapshot(target, DEBERTA)
    (target / "model.safetensors").unlink()
    (target / "model-00001-of-00002.safetensors").write_bytes(b"shard one")
    if broken_shard == "empty":
        (target / "model-00002-of-00002.safetensors").touch()
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "embeddings.weight": "model-00001-of-00002.safetensors",
                    "encoder.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    result = manager.download_model("deberta")

    assert result.state == "failed"
    assert "shard" in (result.message or "").casefold()
    assert calls == []


def test_invalid_existing_directory_is_not_reused_or_removed(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def downloader(**kwargs: Any) -> None:
        calls.append(kwargs)

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))
    target = manager.model_path(DEBERTA)
    target.mkdir(parents=True)
    (target / "partial.bin").write_bytes(b"partial")

    result = manager.download_model("deberta")

    assert result.state == "failed"
    assert "incomplete" in (result.message or "")
    assert calls == []
    assert (target / "partial.bin").is_file()


def test_failed_download_cleans_staging_and_keeps_model_not_ready(tmp_path: Path) -> None:
    def downloader(**kwargs: Any) -> None:
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True)
        (staging / "pytorch_model.bin").write_bytes(b"partial")
        raise RuntimeError("simulated network failure")

    manager = make_manager(tmp_path, downloader, models=(DEBERTA,))

    result = manager.download_model("deberta")

    assert result.state == "failed"
    assert result.message and "download failed" in result.message.casefold()
    assert result.local_dir is not None and not result.local_dir.exists()
    assert not list((tmp_path / "managed-models").rglob(".staging-*"))


def test_refresh_reports_incomplete_local_model_as_not_ready(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, downloader=lambda **_: None, models=(DEBERTA,))
    target = manager.model_path(DEBERTA)
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")

    refreshed = manager.refresh_models()

    assert refreshed[0].state == "failed"
    assert "incomplete" in (refreshed[0].message or "")


def test_explicit_delete_removes_only_selected_managed_model(tmp_path: Path) -> None:
    def downloader(**kwargs: Any) -> None:
        build_valid_snapshot(
            Path(kwargs["local_dir"]), DEBERTA if kwargs["repo_id"] == DEBERTA.model_id else BART
        )

    manager = make_manager(tmp_path, downloader)
    assert manager.download_model("deberta").state == "ready"
    assert manager.download_model("bart").state == "ready"

    deleted = manager.delete_model("deberta")

    assert deleted.state == "not downloaded"
    assert deleted.local_dir is not None and not deleted.local_dir.exists()
    bart_status = next(status for status in manager.list_models() if status.model_key == "bart")
    assert bart_status.state == "ready"


def test_clear_cache_handles_missing_and_existing_directory(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, downloader=lambda **_: None, models=(DEBERTA,))

    missing_result = manager.clear_huggingface_cache()
    assert "no cache" in missing_result.casefold()

    cache_dir = tmp_path / "hf-home" / "hub"
    cache_dir.mkdir(parents=True)
    (cache_dir / "cache-file").write_text("cache", encoding="utf-8")
    cleared_result = manager.clear_huggingface_cache()

    assert "cleared" in cleared_result.casefold()
    assert not cache_dir.exists()
    assert (tmp_path / "hf-home").exists()


def test_unknown_model_key_is_reported() -> None:
    manager = ModelManager(
        ManagerSettings((), Path("models"), Path("cache"), Path("models/transformers-cache"))
    )

    with pytest.raises(KeyError, match="Unknown model key"):
        manager.status("missing")


def test_status_dataclass_uses_documented_lifecycle_states() -> None:
    assert get_type_hints(ModelStatus)["state"] is str
