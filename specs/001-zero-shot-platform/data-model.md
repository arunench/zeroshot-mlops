# Data Model: Zero-Shot Classification Platform

## ModelConfig

One configured model entry from `configs/models.yaml`.

| Field | Type | Required | Rules |
|---|---|---:|---|
| `key` | string | Yes | Unique stable configuration key; safe for use in managed local paths |
| `name` | string | Yes | Human-readable display name |
| `model_id` | string | Yes | Hugging Face repository identifier |
| `revision` | string or null | Baselines: Yes; additional models: No | Baselines use the verified full commit SHAs below, never `main`; additional models may use a branch, tag, or commit; when set, use and record verbatim |
| `local_dir` | string | Yes | Explicit, unique, filesystem-safe relative directory slug; never derive paths directly from `model_id` |

The file MUST define at least `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` and
`facebook/bart-large-mnli` with these pinned revisions and directory mappings:

| `model_id` | `revision` | `local_dir` |
|---|---|---|
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` | `deberta-v3-base-zeroshot-v2.0` |
| `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | `bart-large-mnli` |

These are full Hugging Face commit SHAs verified against the model pages on 2026-11-05.
Adding a model requires configuration only; additional entries may omit `revision`, in
which case they use the `unpinned` path component and have no `model_revision` prediction
field. A mutable branch such as `main` is allowed for additional models only and MUST be
documented as a reproducibility trade-off; prefer immutable commit SHAs. Duplicate keys,
duplicate or unsafe `local_dir` values, missing baseline revisions, missing required
values, unsafe paths, and malformed YAML are configuration errors.

The baseline `configs/models.yaml` entries are:

```yaml
models:
  - key: deberta-v3-base-zeroshot-v2
    name: DeBERTa v3 Zero-Shot v2
    model_id: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
    revision: 8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8
    local_dir: deberta-v3-base-zeroshot-v2.0
  - key: bart-large-mnli
    name: BART Large MNLI
    model_id: facebook/bart-large-mnli
    revision: d7645e127eaf1aefc7862fd59a17a5aa8558b8ce
    local_dir: bart-large-mnli
```

The managed snapshot directory is
`MODEL_DIR/<local_dir>/<revision or "unpinned">`. Resolve `local_dir` explicitly from the
config entry and validate it is a single relative path component. The baseline revision
SHAs are safe path components. For non-SHA revisions, convert the value to a deterministic
filesystem-safe component before joining and prevent traversal outside `MODEL_DIR`; retain
the original configured revision separately for model loading and prediction metadata.
`HF_HOME` is the primary Hub cache variable. `TRANSFORMERS_CACHE` remains a compatibility
setting, also located under the ignored `models/` directory.

## ModelStatus

Runtime view of a configured model; it is not persisted as prediction data.

| Field | Type | Description |
|---|---|---|
| `model_key` | string | Link to `ModelConfig.key` |
| `state` | enum | `not downloaded`, `downloading`, `ready`, or `failed` |
| `revision` | string or null | Configured revision |
| `local_dir` | path or null | `MODEL_DIR/<local_dir>/<revision or "unpinned">` when available |
| `cache_location` | path or null | Relevant Hub cache location when available |
| `message` | string or null | Safe administrator-facing state/failure detail |

`ready` requires all expected model and tokenizer files for the configured revision and
valid model metadata. It does not require successful model loading or test inference.
Partial or failed staging directories are never reported as ready.

## LabelSet

One named set in the `label_sets` collection in `configs/labels.yaml`.

| Field | Type | Required | Rules |
|---|---|---:|---|
| `key` | string | Yes | Unique stable key |
| `name` | string | Yes | Human-readable set name |
| `labels` | array of strings | Yes | At least two; trim whitespace; reject empty or duplicates after case folding |

The user explicitly selects a saved label set for every classification.
`configs/labels.yaml` MAY have a top-level `default_set` key containing the key of one
entry in `label_sets`. The key is optional and MUST reference an existing set when present.
It only supplies a suggested initial selection; it does not select on behalf of the user.
The YAML shape is:

```yaml
label_sets:
  - key: sentiment
    name: Sentiment
    labels:
      - positive
      - neutral
      - negative
default_set: sentiment
```

## ClassificationRequest

Transient input to one classification operation; not persisted independently.

| Field | Type | Required | Rules |
|---|---|---:|---|
| `input_text` | string | Yes | Trim surrounding whitespace; reject empty, over 10,000 characters, or disallowed control characters |
| `model_key` | string | Yes | Must resolve to a model with `ready` status |
| `label_set_key` | string | Yes | Must resolve to a valid saved set explicitly selected by the user |
| `session_id` | string or null | No | Optional, non-secret session context |

Validation occurs before inference. Input is treated as plain text and safely escaped in
the UI. Raw input is not written to logs.

## PredictionRecord

The durable JSON object written to MinIO after successful inference. The full JSON Schema
is [contracts/prediction.schema.json](./contracts/prediction.schema.json).

| Field | Type | Required | Description |
|---|---|---:|---|
| `request_id` | string | Yes | Globally unique request identifier; also used in object key |
| `timestamp` | string | Yes | ISO 8601 UTC timestamp |
| `input_text` | string | Yes | Validated classification input |
| `predicted_class` | string | Yes | Highest-ranked candidate label |
| `confidence` | number | Yes | Score for the predicted class, in `[0, 1]` |
| `candidate_labels` | array of strings | Yes | Labels submitted for inference |
| `ranked_predictions` | array of objects | Yes | All labels with scores, descending by score |
| `model_name` | string | Yes | Configured display name |
| `model_id` | string | Yes | Hugging Face repository identifier |
| `model_revision` | string | No | Configured revision, verbatim; omit when unset |
| `application_version` | string | Yes | Application release identifier |
| `session_id` | string | No | Optional session context |
| `latency_ms` | number | No | Optional non-negative inference latency |

The object key is
`predictions/YYYY/MM/DD/{request_id}.json`, using the UTC date from `timestamp`.
Do not write a success record until MinIO confirms the object write.

`tests/unit/test_model_manager.py`, `tests/unit/test_classifier.py`, and
`tests/unit/test_minio_storage.py` cover core services. The MinIO integration suite is
`tests/integration/test_minio_storage.py`.

## Relationships and lifecycle

- `ModelConfig` has zero or one current `ModelStatus`.
- `LabelSet` is selected by each `ClassificationRequest`.
- A successful request produces exactly one `PredictionRecord`; a validation, inference,
  or storage failure produces no success-shaped result.
- Download lifecycle: `not downloaded` → `downloading` → `ready` or `failed`.
  A retry moves `failed` → `downloading`; explicit deletion returns a ready model to
  `not downloaded`.
