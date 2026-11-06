# Implementation Plan: Zero-Shot Classification Platform

**Branch**: `001-zero-shot-platform` | **Date**: 2026-11-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification and user-provided technical choices in
`specs/001-zero-shot-platform/spec.md`.

## Summary

Build a local, single-machine zero-shot classification learning platform with a
Streamlit multipage UI, separate classification/model/storage/config services, and
MinIO as the prediction system of record. Administrators manage configured Hugging Face
models and candidate-label sets; users select a model and label set, classify text, and
review results. Model snapshots are downloaded into a controlled local directory using
staging and atomic promotion, and the Transformers PyTorch pipeline is cached to avoid
unnecessary model reloads.

The MinIO bucket uses `zeroshot-learning-output` by default because DNS-compatible S3
bucket names reject underscores. This substitution requires a documented ADR before
implementation. See [research.md](./research.md) and
[data-model.md](./data-model.md).

## Technical Context

**Language/Version**: Python 3.11 (project baseline; pin exact patch version in runtime
metadata)

**Primary Dependencies**: PyTorch; Hugging Face Transformers zero-shot-classification
pipeline; Streamlit multipage; MinIO official Python client; PyYAML; python-dotenv;
pytest; Ruff (lint and format)

**Storage**: MinIO object storage for prediction JSON; YAML files at
`configs/models.yaml` and `configs/labels.yaml` for non-secret project configuration;
local `models/` directory for model snapshots and Hugging Face cache. Model locations
resolve to `MODEL_DIR/<local_dir>/<revision or "unpinned">`.

**Testing**: pytest unit tests with mocked model/pipeline and MinIO client; MinIO
integration tests; end-to-end prediction persistence test; GitHub Actions workflow runs
lint, format, unit, MinIO integration, and end-to-end checks on every push and pull request

**Target Platform**: Local Windows/macOS/Linux developer environment; Docker Compose
MinIO service; host application bound to `127.0.0.1` only

**Project Type**: Single Python Streamlit application

**Performance Goals**: Do not repeatedly reload a model during consecutive
classifications. V1 defines no latency or throughput SLA; record optional inference
latency for observation.

**Constraints**: PyTorch inference only; two required models and configuration-only
additions; complete model downloads staged then promoted atomically; one large pipeline
resident at a time; model artifacts and secrets excluded from Git; predictions are
successful only after MinIO write; administrator UI is unauthenticated and loopback-only.

**Scale/Scope**: Single local instance, one administrator and local users, two required
models, configuration-sized label sets, single-prediction JSON objects. No distributed
serving or multi-tenant operation.

## Constitution Check

| Principle / constraint | Plan alignment |
|---|---|
| Learning-platform purpose and v1 boundaries | Local single-instance app; no training, distributed orchestration, or production multi-tenancy |
| Fixed Python/PyTorch/Transformers/Streamlit/MinIO stack | All used as required; no alternate inference backend |
| Reproducibility | Pin Python/dependencies; record model IDs and configured revisions; document MinIO settings |
| Model lifecycle | Download from Hugging Face into staging under `models/`; validate completeness; atomically promote; reuse; explicit deletion only |
| Persistence | Every successful prediction written as JSON to MinIO; YAML is only project configuration |
| Configuration and secrets | Commit non-secret YAML; load credentials from environment or secret management; `.env.example` placeholders only |
| Separation of concerns | Streamlit pages are UI-only; model, classifier, MinIO, configuration, and logging live in separate services/utilities |
| Testing and quality gates | pytest, mocked large models, MinIO integration tests, CI lint/format/unit/integration/end-to-end checks, and final acceptance verification |
| Error handling, observability, and privacy | Clear service errors; structured events; avoid logging input text or credentials |
| Local administrator boundary | Bind Streamlit to loopback; explicitly state no authentication and local trust boundary |
| ADR process | Document the DNS-compatible bucket substitution before implementation |

**Gate result**: PASS, subject to adding the required bucket-name ADR before implementation
work begins. No stack exception or unresolved constitution conflict is proposed.

## Architecture and Components

The Streamlit entry point registers two pages and delegates all non-UI behavior to
services. Configuration is loaded once from YAML plus environment variables. The model
manager owns snapshot downloads, readiness validation, reuse, and explicit cleanup. The
classifier owns one active Transformers pipeline at a time and produces normalized
ranked predictions. The MinIO service creates/checks the configured bucket and writes
the required JSON object before the UI reports success. Structured logging is configured
centrally and redacts secrets and avoids recording raw input.

### Component responsibilities

- `app/streamlit_app.py`: configure the local multipage application, loopback-safe
  startup guidance, shared page setup.
- `app/pages/1_Admin.py`: model lifecycle and label-set controls only; call services,
  display progress/status, and require explicit confirmation for deletion.
- `app/pages/2_Classify_Text.py`: collect user text/model/label selection, display
  validation and service errors, and render successful results.
- `app/services/model_manager.py`: configured-model discovery, mapping each configured
  model ID through its explicit `local_dir`, revision-aware local path resolution,
  staged download, completeness/metadata validation, ready status, refresh, removal,
  and Hugging Face cache cleanup.
- `app/services/classifier.py`: lazily create and cache the PyTorch Transformers
  zero-shot pipeline for a single active model; release/switch a previous large model
  deliberately; rank all label scores and return timing/model metadata.
- `app/services/minio_storage.py`: MinIO client construction, bucket existence/create,
  unique date-partitioned object path, and JSON persistence.
- `app/utils/config.py`: dotenv/environment loading, YAML validation, typed settings,
  safe defaults, and no secret fallback values.
- `app/utils/logging_config.py`: structured log setup and secret/input-safe fields.

### Streamlit page design

The Admin page is split into model management and candidate-label management. The model
table shows configured ID/revision, status, local path, and cache path, with download,
refresh, model delete, and cache cleanup actions. Downloads show per-model progress and
failures. Destructive actions require confirmation. Label-set controls add/edit/remove
labels, mark an optional default, and show validation errors.

The Classify page requires explicit model and candidate-label selection. A configured
default label set may initialize the control but does not bypass the user's selection.
Input is trimmed, length-checked (10,000 characters maximum), and checked for disallowed
control characters before inference. Results show predicted class, confidence, all labels
ranked by score, model, and timestamp. The page shows success only after MinIO persistence
completes.

### Model download, readiness, and caching

1. Resolve the configured model identifier, required pinned revision for each baseline
   model, and explicit local directory from `configs/models.yaml`.
2. Download into a unique temporary directory on the same filesystem as the final model
   directory, passing the configured revision explicitly.
3. Validate required model/tokenizer files and metadata; failures leave the final
   directory untouched and report `failed`.
4. Atomically rename the complete staging directory into
   `MODEL_DIR/<local_dir>/<revision or "unpinned">`. Use a filesystem-safe revision
   component; the baseline full commit SHAs are already safe path segments. A complete
   existing directory is reused without downloading.
5. Mark `ready` only when the expected files and metadata are valid; model loading and
   test inference are intentionally not part of readiness.
6. On explicit deletion, release an active pipeline for that model before removing only
   the managed model directory. Cache cleanup is a separate explicit action and handles
   an absent cache path as a successful no-op with an informative status.

The classifier caches a single active pipeline keyed by model identifier and revision.
Repeated use of that model reuses the pipeline. Switching models releases the previous
large model before loading the next, avoiding unnecessary multiple model allocations.
No inference backend fallback is permitted.

### Hugging Face integration and revision recording

The configured revision is passed to the Transformers/Hugging Face load operation and
shown in model status. Both baseline models have full commit SHA revisions pinned in
`configs/models.yaml`; their values are included below and recorded verbatim as
`model_revision` in predictions. These baseline SHAs are verified against their Hugging
Face model pages and MUST NOT be replaced with `main` or invented values. Additional
model entries may use a branch, tag, or commit, or omit revision; a mutable branch is a
reproducibility trade-off and MUST be documented as such. Entries without revision are
stored under `unpinned`, shown as unpinned, and omit `model_revision` from records.

| Model ID | Pinned revision | `local_dir` |
|---|---|---|
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` | `deberta-v3-base-zeroshot-v2.0` |
| `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | `bart-large-mnli` |

Map model IDs to local paths using the explicit `local_dir` value in each model
configuration entry. These stable slugs avoid deriving filesystem paths from a
slash-separated Hugging Face ID and make collisions and path traversal detectable.

### MinIO integration

Use the official MinIO Python client with endpoint, access key, secret key, secure flag,
and bucket from environment configuration. At startup or before the first write, check
the bucket and create it if missing; surface creation/write errors. The default is
`zeroshot-learning-output`, and a required ADR documents substituting it for the original
underscore name. Use UTC timestamp partitioning and a collision-resistant request ID for
`predictions/YYYY/MM/DD/{request_id}.json`. Serialize the
[PredictionRecord](./data-model.md#predictionrecord) according to
[prediction.schema.json](./contracts/prediction.schema.json).

### Environment variables

| Variable | Purpose | Default / handling |
|---|---|---|
| `MINIO_ENDPOINT` | MinIO host and port | Required |
| `MINIO_ACCESS_KEY` | MinIO access key | Required; secret |
| `MINIO_SECRET_KEY` | MinIO secret key | Required; secret |
| `MINIO_SECURE` | Use TLS for MinIO | `false` for local Compose |
| `MINIO_BUCKET` | Prediction bucket | `zeroshot-learning-output` |
| `MODEL_DIR` | Managed downloaded model location | `models/downloaded` |
| `HF_HOME` | Hugging Face cache root | `models/hf-home` |
| `TRANSFORMERS_CACHE` | Transformers cache root | `models/transformers-cache` |
| `HF_TOKEN` | Optional Hugging Face token for gated/private models | Optional; secret |
| `APP_VERSION` | Version recorded with predictions | Project version |

For an application running directly on the host, set `MINIO_ENDPOINT=localhost:9000`.
For an application running in Docker Compose, set `MINIO_ENDPOINT=minio:9000` so it uses
the Compose service DNS name and internal port. Do not use `localhost` between containers.

Use python-dotenv to load a developer `.env` file when present, without overwriting
explicitly exported environment values. `.env.example` contains placeholders only. Never
log secret values; do not silently substitute credentials or a different inference
backend.

`HF_HOME` is the primary Hugging Face cache location. `TRANSFORMERS_CACHE` is retained
for compatibility with libraries or versions that still consult it; keep it pointed
under the managed `models/` area and do not treat it as the primary cache setting.

### Configuration files

- `configs/models.yaml`: stable model key, display name, Hugging Face identifier, pinned
  `revision`, and managed local directory name. Include both baseline model entries with
  the exact IDs, revisions, and `local_dir` slugs in the table above. Additional models
  require only a configuration change and may omit `revision`.
- `configs/labels.yaml`: a `label_sets` collection with saved set names and label arrays,
  plus an optional top-level `default_set` key referencing one set. Admin edits are
  validated and saved atomically to the same file; default selection only suggests an
  initial set and users still choose the set they use.

Configuration YAML is non-secret and may be committed. Reject malformed or duplicate
model keys, absent required fields, invalid default references, and invalid label sets
with actionable messages; never replace malformed files with empty defaults.

### Testing strategy

- Unit tests live in `tests/unit/test_model_manager.py`,
  `tests/unit/test_classifier.py`, and `tests/unit/test_minio_storage.py`. They cover
  configuration parsing/validation, safe model path mapping, atomic promotion and
  cleanup, ready-state validation, revision forwarding, score sorting, text validation,
  model-pipeline reuse/switch behavior, unique object naming, and error handling.
- Mock `transformers.pipeline`, model downloads, filesystem effects where practical, and
  MinIO clients in fast unit tests. Do not download large models in ordinary CI.
- MinIO integration tests live in `tests/integration/test_minio_storage.py`, are
  configured against a running MinIO service, and verify bucket auto-creation and
  prediction JSON/path.
- End-to-end local validation starts the stack, downloads a model, classifies, inspects
  the object, explicitly deletes a model or clears cache, and runs tests.
- GitHub Actions runs on every push and pull request. Each CI run performs Ruff lint and
  format checks, unit tests, MinIO integration tests, and the end-to-end
  prediction-to-MinIO test. The workflow starts its own MinIO container, waits for its
  readiness endpoint, and supplies non-secret CI-only MinIO credentials through workflow
  environment variables. The end-to-end test uses a mocked Transformers pipeline, so CI
  downloads no models. CI runs the full test suite rather than making MinIO coverage
  conditional.

### Docker and local development

Docker Compose starts a local MinIO service with persistent named volume, health check,
and credentials sourced from environment values without usable hard-coded fallbacks. The
application Dockerfile uses a pinned Python base, installs locked dependencies, excludes
`models/`, `.env`, and caches from the image context, and runs as a non-root user. A
host-run app binds to `127.0.0.1` using `.streamlit/config.toml` and the documented
`python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1` command. In
Compose, Streamlit listens on the container interface (`0.0.0.0`) while the published host
port is bound only to loopback as `127.0.0.1:8501:8501`. This enforces local
unauthenticated administrator access; verify the app is not reachable from other devices
and do not expose administrator controls to the network. A developer can also run the app
directly in a Python virtual environment.
Makefile targets provide common setup, lint, test, MinIO, app, and cleanup commands.
Every Python tool is also invoked as a module so Windows does not depend on potentially
blocked `.exe` shims. The Makefile is optional; the equivalent direct commands are:

| Action | Makefile target | Direct command |
|---|---|---|
| Run app on host | `make app` | `python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1` |
| Lint | `make lint` | `python -m ruff check .` |
| Check formatting | `make format-check` | `python -m ruff format --check .` |
| Unit tests | `make test` | `python -m pytest tests/unit` |
| MinIO integration tests | `make test-integration` | `python -m pytest tests/integration` |
| Start MinIO | `make minio-up` | `docker compose up -d minio` |
| Stop MinIO | `make minio-down` | `docker compose down` |

Use the Docker Compose app target for a Compose-hosted application, configured with
`MINIO_ENDPOINT=minio:9000`; see [quickstart.md](./quickstart.md) for full setup.

### Deployment approach

V1 targets local single-machine demonstrations only. Host-run Streamlit binds to
`127.0.0.1`. In Docker Compose, Streamlit listens on `0.0.0.0` inside its container so
Docker can forward requests, but its host port mapping MUST be
`127.0.0.1:8501:8501`; MinIO ports are likewise restricted to host loopback. Acceptance
verification confirms another device on the local network cannot reach the application.
This is not a production or multi-user deployment. No public ingress, remote administrator
access, or production authentication is included.

## Project Structure

### Documentation (this feature)

```text
specs/001-zero-shot-platform/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── prediction.schema.json
└── tasks.md
docs/
└── adr/
    └── 0001-minio-bucket-name.md
```

### Source Code

```text
app/
├── streamlit_app.py
├── pages/
│   ├── 1_Admin.py
│   └── 2_Classify_Text.py
├── services/
│   ├── model_manager.py
│   ├── classifier.py
│   └── minio_storage.py
└── utils/
    ├── config.py
    └── logging_config.py
configs/
├── models.yaml
└── labels.yaml
models/
└── .gitkeep
tests/
├── unit/
│   ├── test_model_manager.py
│   ├── test_classifier.py
│   └── test_minio_storage.py
├── integration/
│   └── test_minio_storage.py
└── e2e/
.github/
└── workflows/
    └── ci.yml
.streamlit/
└── config.toml
Dockerfile
docker-compose.yml
Makefile
pyproject.toml
requirements.lock
.env.example
.dockerignore
.gitignore
README.md
```

**Structure Decision**: Use one Python project because the UI and backend services are a
single local application with no separate deployment boundary. Keep Streamlit pages
presentation-only; put lifecycle, inference, persistence, settings, and logging behind
separate modules. This follows the constitution's separation-of-concerns principle
without introducing a second application or service.

The repository `.gitignore` MUST ignore `models/*` while explicitly allowing
`!models/.gitkeep`. This keeps downloaded snapshots and caches out of Git while preserving
the empty managed-model directory.

## Complexity Tracking

No constitution violations or additional architectural layers are proposed. The bucket
name substitution is a documented ADR prerequisite, not a waiver of governance.
