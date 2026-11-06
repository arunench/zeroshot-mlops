# Tasks: Zero-Shot Classification Platform

**Input**: Design documents from `specs/001-zero-shot-platform/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and
`contracts/prediction.schema.json`

**Organization**: Tasks are grouped by user story after shared setup and foundational
work so each story has a stated independent validation path.

## Phase 1: Setup

**Purpose**: Establish the required architecture decision and reproducible project
scaffolding. The bucket-name ADR is required before application implementation begins.

- [ ] T001 Document the substitution of `zeroshot-learning-output` for `zeroshot_learning_output`, including reason, benefits, risks, and reproducibility impact, in `docs/adr/0001-minio-bucket-name.md`.
- [ ] T002 Define Python 3.11 support with an exact patch pin, the CPU build of PyTorch, exact Transformers, Streamlit, and MinIO client versions plus pinned compatible dependencies, Ruff settings, and pytest configuration in `pyproject.toml`; record the fully resolved dependency set in `requirements.lock`.
- [ ] T003 Ignore `.env`, Python/build outputs, model artifacts and caches while allowing `!models/.gitkeep` in `.gitignore`; create `.dockerignore` and `models/.gitkeep`.
- [ ] T004 Add both required model IDs, their explicit local directory slugs, and full commit SHA revisions verified against their Hugging Face model pages to `configs/models.yaml` (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`: `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`; `facebook/bart-large-mnli`: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`); never use `main` or invented baseline revisions; add a valid sample label set and optional top-level `default_set` to `configs/labels.yaml`.
- [ ] T005 Add a MinIO Docker Compose service with a pinned image, persistent volume, health check, and loopback-only host port bindings in `docker-compose.yml`; require credentials to come from environment variables with no usable hard-coded fallbacks, and add placeholder-only environment values in `.env.example`.
- [ ] T006 Add optional Makefile targets for host app startup bound to `127.0.0.1`, lint, formatting, unit tests, integration tests, and MinIO lifecycle in `Makefile`; list the equivalent `python -m` commands in the file.

## Phase 2: Foundational

**Purpose**: Implement and verify shared configuration, logging, and MinIO services that
block the user workflows.

- [ ] T007 [P] Add tests for required environment settings, safe defaults, YAML parsing and validation, malformed configuration errors, and secret-safe settings in `tests/unit/test_config.py`.
- [ ] T008 Implement typed environment and YAML configuration loading for `configs/models.yaml` and `configs/labels.yaml` in `app/utils/config.py`; enforce a unique `ModelConfig.key`, unique `local_dir` as a single filesystem-safe relative path component, and the exact verified baseline commit SHAs specified in `data-model.md` (never use `main` or invented baseline revisions); allow additional models to use a branch, tag, or commit, and document mutable-branch use as a reproducibility trade-off; apply label rule “At least two; trim whitespace; reject empty or duplicates after case folding”; set `HF_HOME` and `TRANSFORMERS_CACHE` in the process environment before any `transformers` or `huggingface_hub` imports; support `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_SECURE`, `MINIO_BUCKET`, `MODEL_DIR`, `HF_TOKEN`, and `APP_VERSION`, using `zeroshot-learning-output` as the bucket default.
- [ ] T009 Configure structured logging without raw classification text or credential values in `app/utils/logging_config.py`.
- [ ] T010 [P] Add unit tests for bucket creation and MinIO failures in `tests/unit/test_minio_storage.py`; verify `request_id` is unique, `timestamp` is ISO 8601 UTC, `confidence` and every ranked `score` are in `[0, 1]`, and objects use the UTC date-based path.
- [ ] T011 Implement MinIO client setup, automatic bucket creation, and `PredictionRecord` JSON writes to `predictions/YYYY/MM/DD/{request_id}.json` in `app/services/minio_storage.py`; require globally unique `request_id`, ISO 8601 UTC `timestamp`, `confidence` and ranked scores in `[0, 1]`, and fields `input_text`, `predicted_class`, `candidate_labels`, `ranked_predictions`, `model_name`, `model_id`, and `application_version`; preserve configured `model_revision` verbatim and omit it when unset; include optional `session_id` and non-negative `latency_ms` when available; log structured writes and return success only after MinIO confirms the write.
- [ ] T012 Create the Streamlit multipage entry point and shared page setup in `app/streamlit_app.py`; set `HF_HOME` and `TRANSFORMERS_CACHE` in the process environment before importing any `transformers` or `huggingface_hub` modules, and configure `.streamlit/config.toml` with `server.address = "127.0.0.1"` for host-run startup through `python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1`.

## Phase 3: User Story 1 - Manage Model Availability (Priority: P1)

**Goal**: Let the administrator view, download, reuse, refresh, and explicitly remove
models and Hugging Face cache files.

**Independent Test**: With model downloads mocked, verify both configured models are
listed, staged downloads become ready only after required model/tokenizer files and
metadata validate, complete local revisions are reused, failed/partial downloads are not
ready, and explicit deletion and missing-cache cleanup update status correctly.

### Tests for User Story 1

- [ ] T013 [P] [US1] Add unit tests for model configuration mapping, revision-qualified paths, staged download and promotion, reuse, failures, explicit deletion, refresh, and missing Hugging Face cache cleanup in `tests/unit/test_model_manager.py`; verify status is exactly `not downloaded`, `downloading`, `ready`, or `failed`, readiness requires expected model/tokenizer files and valid metadata, and sharded-weight index validation rejects a missing or empty shard file.

### Implementation for User Story 1

- [ ] T014 [US1] Implement configured model discovery, safe `MODEL_DIR/<local_dir>/<revision or "unpinned">` paths, deterministic filesystem-safe conversion for non-SHA revisions while preserving the original configured revision, and staged Hugging Face downloads using explicit `allow_patterns` for PyTorch-loadable weights (`*.safetensors`, `pytorch_model*.bin`), sharded-weight indexes (`model.safetensors.index.json`, `pytorch_model.bin.index.json`), model configuration (`config.json`, `generation_config.json`), and tokenizer files (tokenizer metadata, vocabularies, merges, and SentencePiece files), with `ignore_patterns` excluding TensorFlow, Flax, ONNX, and other non-PyTorch weights; define readiness against the same patterns and require valid model metadata, at least one PyTorch-loadable weight file, the matching index file whenever weights are sharded, and the tokenizer files required by each configured model; when either sharded-weight index is present, parse it and verify every named shard file exists and is non-empty before marking the model ready or reusing it; implement refresh, explicit deletion, and cache cleanup; expose only `not downloaded`, `downloading`, `ready`, or `failed` states and log structured lifecycle events.
- [ ] T015 [US1] Implement administrator model listing, status/location display, multi-model download, refresh, and confirmation-gated model/cache deletion controls in `app/pages/1_Admin.py`.

## Phase 4: User Story 3 - Classify Text and Review Results (Priority: P1)

**Goal**: Let a user select an available model and saved label set, submit validated text,
and view the ranked classification only after persistence succeeds.

**Independent Test**: With a ready fixture model, saved label set, mocked Transformers
pipeline, and MinIO test double, verify validation occurs before inference, scores include
all labels in descending order, a prediction is saved, and model, input, or storage errors
are shown without success-shaped results.

### Tests for User Story 3

- [ ] T016 [P] [US3] Add unit tests for `input_text` rule “Trim surrounding whitespace; reject empty, over 10,000 characters, or disallowed control characters”; verify at least two valid labels, PyTorch pipeline selection, and that simulated PyTorch unavailability produces a clear inference error without selecting another backend; verify confidence and scores in `[0, 1]`, all labels ranked by descending score, model reuse/switching, prediction metadata, and failure handling in `tests/unit/test_classifier.py`.

### Implementation for User Story 3

- [ ] T017 [US3] Implement validated zero-shot inference with the Transformers pipeline explicitly set to the PyTorch backend, one-active-model pipeline reuse, score normalization to `[0, 1]` and descending ranking of every label, latency measurement, and prediction metadata in `app/services/classifier.py`; trim input and reject empty, over-10,000-character, or disallowed-control-character input and fewer than two labels before inference; log structured inference events without raw text.
- [ ] T018 [US3] Implement the classification workflow in `app/pages/2_Classify_Text.py`; require a ready model and explicit saved-label-set selection, show clear errors for invalid input, fewer than two labels, missing models, and storage failures, escape user input and classification results for safe plain-text display, and display class, confidence, ranked labels, model, and timestamp only after MinIO persistence succeeds.

## Phase 5: User Story 4 - Persist and Reproduce Platform Results (Priority: P1)

**Goal**: Verify the MinIO persistence contract and provide the documented local workflow
for starting services, performing a classification, inspecting its JSON, and running tests.

**Independent Test**: With local MinIO running, exercise the prediction write, confirm
automatic bucket creation and exactly one JSON object at the UTC date-based request path
with all required fields, then complete the documented cleanup and test workflow.

### Tests for User Story 4

- [ ] T019 [P] [US4] Add MinIO integration tests for automatic bucket creation, the configured bucket, JSON content type, required prediction fields, and `predictions/YYYY/MM/DD/{request_id}.json` paths in `tests/integration/test_minio_storage.py`; assert unique `request_id`, ISO 8601 UTC `timestamp`, and `confidence`/ranked scores in `[0, 1]`.
- [ ] T020 [P] [US4] Add an end-to-end test that uses a mocked large-model pipeline and live MinIO to classify valid input, write and inspect exactly one JSON object with `request_id`, `timestamp`, `input_text`, `predicted_class`, `confidence`, `candidate_labels`, `ranked_predictions`, `model_name`, `model_id`, and `application_version`, verify configured `model_revision` is verbatim or omitted when unset, and report a failed write as unsuccessful in `tests/e2e/test_prediction_to_minio.py`.

### Implementation for User Story 4

- [ ] T021 [US4] Document prerequisites, `.env` setup, host and Compose MinIO endpoints, app startup, model download, classification, JSON inspection, explicit cleanup, lint, unit, integration, and end-to-end test commands in `README.md`.

## Phase 6: User Story 2 - Manage Candidate Label Sets (Priority: P2)

**Goal**: Let an administrator add, edit, remove, save, and optionally suggest a default
saved label set while requiring the user to select the set used for classification.

**Independent Test**: Save a valid set, restart/reload configuration and confirm it remains
available; reject sets with fewer than two labels, empty labels, or duplicates after
trimming and case folding; verify removing a set updates configuration and invalidates a
matching `default_set`.

### Tests for User Story 2

- [ ] T022 [P] [US2] Add tests for saved label-set creation, editing, removal, persistence, optional top-level `default_set`, and label rule “At least two; trim whitespace; reject empty or duplicates after case folding” in `tests/unit/test_config.py`.

### Implementation for User Story 2

- [ ] T023 [US2] Implement label-set add/edit/remove/save operations and validation in `app/utils/config.py`; apply label rule “At least two; trim whitespace; reject empty or duplicates after case folding”, persist `label_sets` and optional top-level `default_set` atomically to `configs/labels.yaml`, and retain no per-set `is_default` field.
- [ ] T024 [US2] Add administrator label-set controls to `app/pages/1_Admin.py`, including validation feedback, save/remove confirmation, and optional default selection without bypassing the user's later selection.

## Phase 7: Polish and Cross-Cutting Concerns

**Purpose**: Complete automated quality gates, containerized app support, and acceptance
validation across stories.

- [ ] T025 [P] Configure `.github/workflows/ci.yml` to run on every push and pull request and execute Ruff lint, Ruff format checks, unit tests, MinIO integration tests, and `tests/e2e/test_prediction_to_minio.py`; start a MinIO container in the workflow, wait for its health endpoint before tests, and pass disposable non-secret CI-only MinIO credentials to the app/test process through workflow `env` variables; ensure the end-to-end test mocks the Transformers pipeline and downloads no model.
- [ ] T026 Add a pinned Python base image, locked dependency installation, non-root runtime, and model/cache/secret exclusions to `Dockerfile`.
- [ ] T027 Add an optional app service to `docker-compose.yml`, configured to reach MinIO at `minio:9000`; run Streamlit with `--server.address 0.0.0.0` inside the container and publish the app as `127.0.0.1:8501:8501` so the host port is loopback-only.
- [ ] T028 Verify the README demo workflow end to end: start MinIO, download a model, classify, inspect the MinIO JSON, explicitly delete a model or clear cache, and run tests; document host-run binding to `127.0.0.1` versus Compose binding to container `0.0.0.0` with host mapping `127.0.0.1:8501:8501`, and verify the app is not reachable from other devices; update discrepancies in `README.md`.
- [ ] T029 Verify Ruff lint and format checks, unit tests, MinIO integration tests, and the mocked-pipeline end-to-end prediction-to-MinIO test; perform the final acceptance-criteria verification; verify acceptance criterion 18 by confirming host-run Streamlit binds to `127.0.0.1`, Compose publishes only `127.0.0.1:8501:8501`, and the app is unreachable from other devices on the local network; inspect `.gitignore`, `.dockerignore`, `.env.example`, `docker-compose.yml`, `models/`, and tracked repository files to confirm no `.env`, credentials, usable hard-coded credentials, model artifacts, or model caches are committed; confirm the constitution, `spec.md`, `plan.md`, `data-model.md`, and `tasks.md` are present.

## Dependencies and Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 establishes the required bucket-name ADR before implementation; remaining setup tasks establish manifests, configuration, and local infrastructure.
- **Foundational (Phase 2)**: Depends on Setup; shared configuration, logging, storage, and app entry point block user workflows.
- **User stories (Phases 3-6)**: Depend on the Foundational phase. US3 uses the model manager from US1; US2 extends the Admin page created for US1. US3 can be independently tested with a preconfigured saved label set before US2's administration UI is complete.
- **Polish (Phase 7)**: Depends on all desired stories; the demo and final gates require integrated implementation.

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational; no dependency on another user story.
- **US3 (P1)**: Starts after US1 for model availability; uses seeded label configuration and can be unit-tested with fixtures. The full administrator-to-user workflow also requires US2.
- **US4 (P1)**: Starts after Foundational; storage integration and end-to-end persistence can be tested independently of the administrator workflows.
- **US2 (P2)**: Starts after Foundational and follows US1 because its controls extend the same `app/pages/1_Admin.py` file.

### Parallel Opportunities

- T003, T004, and T005 modify distinct setup files and can proceed in parallel after T001.
- T007 and T010 create independent configuration and MinIO unit-test files; they can proceed in parallel.
- T013 and T016 create tests in separate files and can be prepared in parallel after the foundation.
- T019 and T020 are separate persistence test files and can be prepared in parallel once the MinIO contract is established.
- T025 can proceed independently from T026 after application/test commands and the dependency lockfile are established.

### Parallel Example: Shared Foundation

```text
Task: T007 Add configuration tests in tests/unit/test_config.py
Task: T010 Add MinIO storage tests in tests/unit/test_minio_storage.py
```

### Parallel Example: Independent Story Tests

```text
Task: T013 Add model lifecycle tests in tests/unit/test_model_manager.py
Task: T016 Add classifier tests in tests/unit/test_classifier.py
```

## Implementation Strategy

### MVP First

1. Complete Setup and Foundational phases, including the bucket-name ADR.
2. Complete US1 to demonstrate safe model download, readiness, reuse, and explicit cleanup.
3. Stop and validate US1 independently with mocked downloads before proceeding.

### Incremental Delivery

1. Add US3 classification against a ready model, saved labels, and MinIO persistence.
2. Add US4 live MinIO and end-to-end verification plus the reproducible developer workflow.
3. Add US2 administrator label-set management and validate the full admin-to-user workflow.
4. Complete container, CI, security, and final acceptance checks.

## Completion Criteria

- The documented demo covers setup, bucket creation, model management, classification,
  MinIO JSON verification, explicit cleanup, and test execution.
- The unit, MinIO integration, end-to-end, lint, and format checks pass.
- The repository contains no committed credentials, `.env` files, model artifacts, or
  model cache contents.
- Every task above follows the checklist format `- [ ] T### [P?] [Story?] description`
  and names the file path(s) it changes.
