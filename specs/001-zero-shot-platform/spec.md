# Feature Specification: Zero-Shot Classification Platform

**Feature Branch**: `001-zero-shot-platform`

**Created**: 2026-11-05

**Status**: Draft

**Input**: User description: "Build a zero-shot text classification platform with an administrator workflow and a user workflow, model management, candidate-label management, classification, and MinIO prediction persistence."

## Purpose and Scope

V1 provides a local learning platform for zero-shot text classification, including
administrator model and candidate-label management, user classification, and durable
prediction records. It excludes model training or fine-tuning, distributed orchestration,
production authentication, and multi-tenant operation.

## Clarifications

### Session 2026-11-05

- Q: Which bucket name should the platform use by default if MinIO rejects `zeroshot_learning_output` because of its naming rules? → A: Use `zeroshot-learning-output` as the default bucket and document the substitution in an ADR.
- Q: What must be true before the administrator can see a model marked `ready`? → A: All expected model and tokenizer files for the configured revision must be present, and model metadata must pass validation; loading or test inference is not required for readiness.
- Q: How should the platform record a configured model revision in each prediction? → A: Record the configured revision string verbatim; omit it when none is configured.
- Q: How should the unauthenticated administrator workflow be limited to local access? → A: Bind the application to loopback only; do not expose administrator controls to other devices.

## Users and Roles

- **Administrator**: Manages configured models, downloads and deletes local models,
  clears the Hugging Face cache, and creates, edits, removes, saves, and optionally marks
  a candidate-label set as the default.
- **User**: Selects a saved candidate-label set and an available model, submits text, and
  reviews the classification result.
- **Project Maintainer**: Starts and verifies the local application and storage service,
  runs quality checks, and maintains reproducible setup and troubleshooting instructions.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Manage Model Availability (Priority: P1)

An administrator reviews the configured zero-shot models, downloads selected models, checks
their availability, and removes models or cached files only through explicit cleanup actions.
This makes model setup and reuse visible and manageable before users classify text.

**Why this priority**: Classification depends on a complete, available model. Administrators
need to establish model availability before the other workflows can deliver their value.

**Independent Test**: From an empty model directory, an administrator can view the two required
models, download each, observe its status and location, refresh the list, and verify a second
download request reuses the already available model.

**Acceptance Scenarios**:

1. **Given** no models are downloaded, **When** an administrator opens model management,
   **Then** the configured list includes `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` and
   `facebook/bart-large-mnli`, with each model's identifier and optional configured
   revision.
2. **Given** one or more configured models are selected, **When** the administrator requests
   a download, **Then** each model reports `downloading` and then either `ready` or `failed`,
   and a ready model shows its local model directory and cache location.
3. **Given** a complete model already exists locally, **When** the administrator requests
   its download again, **Then** the existing model is reused without downloading it again.
4. **Given** expected model or tokenizer files are missing, or model metadata fails
   validation, **When** the model status is refreshed, **Then** it is not shown as ready
   and the administrator receives a clear failure message.
5. **Given** a downloaded model is selected, **When** the administrator explicitly confirms
   deletion, **Then** that model's local files are removed and its status is no longer ready.
6. **Given** the Hugging Face cache directory is missing, **When** the administrator requests
   cache cleanup, **Then** cleanup completes safely and reports that no cache was present.

---

### User Story 2 - Manage Candidate Label Sets (Priority: P2)

An administrator creates, edits, removes, saves, and selects named candidate-label sets so
users can choose relevant labels when classifying text. The administrator may optionally
mark one saved set as the default; users still select a saved set in their classification
workflow.

**Why this priority**: Candidate labels define the choices against which text is classified.
Persisted sets let the administrator configure them once and reuse them.

**Independent Test**: An administrator creates and saves a set with at least two unique,
non-empty labels, selects it, restarts the application, and confirms the saved set remains
available. Invalid sets are rejected with an explanation.

**Acceptance Scenarios**:

1. **Given** the administrator enters a set with at least two distinct non-empty labels,
   **When** they save it, **Then** the set is persisted in the configured candidate-label
   configuration file and is available for selection.
2. **Given** a label set contains fewer than two labels, an empty label, or duplicate labels,
   **When** the administrator saves it, **Then** it is rejected and the validation message
   identifies the problem.
3. **Given** one or more saved sets exist, **When** the administrator selects a set,
   **Then** that set is available for users to select during classification.
4. **Given** a saved set is selected, **When** the administrator removes it and confirms,
   **Then** it is removed from the persisted configuration and is no longer selectable.
5. **Given** saved sets exist, **When** the administrator optionally marks one set as
   the default, **Then** it is suggested as the initial selection but users can select
   another saved set.

---

### User Story 3 - Classify Text and Review Results (Priority: P1)

A user enters text, chooses a downloaded model and saved label set, and submits a
classification. The user receives the predicted class, its confidence, and a complete
ranked list of candidate labels and scores.

**Why this priority**: Classification is the platform's central user value and must work
end to end with configured models and labels.

**Independent Test**: With at least one ready model, a saved set of two or more labels, and
available storage, submit non-empty text and verify the result includes a predicted class,
confidence, all label scores in rank order, model identity, and timestamp.

**Acceptance Scenarios**:

1. **Given** a ready model and a valid saved label set, **When** the user submits non-empty
   text, **Then** the result shows the predicted class, confidence, every candidate label
   with its score in descending order, the model used, and a timestamp.
2. **Given** the user submits empty or whitespace-only text, **When** classification is
   requested, **Then** no classification is recorded and the user sees a clear validation
   message.
3. **Given** no model is downloaded, **When** the user opens or submits the classification
   workflow, **Then** classification is unavailable and the user sees a clear message.
4. **Given** no valid label set with at least two labels is available, **When** the user
   attempts classification, **Then** the user sees a clear message and no prediction is
   recorded.
5. **Given** the selected text exceeds the configured maximum or contains disallowed
   control characters, **When** classification is submitted, **Then** the user sees a
   clear validation error and no prediction is recorded.
6. **Given** storage is unavailable, **When** a classification is submitted, **Then** the
   user sees a clear storage error and is not shown a successful, persisted classification.

---

### User Story 4 - Persist and Reproduce Platform Results (Priority: P1)

A project maintainer starts the application and its local storage service using documented
instructions, and can verify that each successful classification is durably recorded with
the metadata needed to understand and reproduce the result.

**Why this priority**: Durable, inspectable results and repeatable setup demonstrate the
platform's MLOps purpose and are necessary for trustworthy end-to-end behavior.

**Independent Test**: Start the documented local services, complete a classification, and
verify that one JSON prediction object appears in the configured bucket at the date-based
path with all required fields. Repeat with storage unavailable and verify the failure is
reported rather than silently discarded.

**Acceptance Scenarios**:

1. **Given** the configured MinIO service is reachable, **When** the application starts and
   the configured bucket is missing, **Then** the bucket is created automatically.
2. **Given** a classification succeeds, **When** its result is persisted, **Then** exactly
   one JSON object is written at
   `predictions/YYYY/MM/DD/{request_id}.json` using the result's date and unique request ID.
3. **Given** a prediction object is written, **When** it is inspected, **Then** it contains
   the request ID, timestamp, input text, predicted class, confidence, candidate labels,
   ranked predictions, model name and Hugging Face identifier, application version, and
   the configured model revision verbatim when present. `model_revision` is omitted when
   no revision is configured; session ID and latency may also be included when available.
4. **Given** a new developer follows the README, **When** they use the documented commands
   and prerequisites, **Then** they can start the local platform and run its tests without
   committing secrets or model files.

### Edge Cases

- A model download is interrupted, leaves partial files, or fails validation; it remains
  unavailable and can be retried without being mistaken for a ready model.
- One selected model fails to download while another succeeds; each model reports its own
  result, and a failed model does not invalidate a complete one.
- A model directory exists but is incomplete or corrupt; it is not reused as ready.
- The model list is refreshed during or after a download or deletion; displayed status and
  local locations reflect the current state.
- Cache cleanup is requested when the Hugging Face cache directory does not exist.
- Labels differ only by surrounding whitespace or letter case; duplicates are rejected
  after trimming and case-insensitive comparison.
- Text is empty after trimming, exceeds the maximum length, or contains disallowed control
  characters; it is rejected with a clear validation message before inference.
- The candidate-label configuration file is missing or malformed; the administrator gets
  a clear error and existing user classification is not falsely reported as successful.
- MinIO is unreachable, rejects a write, or the bucket cannot be created; no successful
  persisted prediction is reported.
- Two requests occur on the same day; unique request IDs prevent object-name collisions.
- User text or credentials appear in diagnostic context; logs do not unnecessarily expose
  text or reveal credentials.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The platform MUST provide separate administrator and user workflows.
- **FR-002**: The administrator MUST be able to view configured zero-shot models, including
  `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` and `facebook/bart-large-mnli`, as defined
  in `configs/models.yaml`.
- **FR-003**: Additional zero-shot models MUST be addable through configuration only,
  by editing `configs/models.yaml` and without application code changes.
- **FR-004**: The administrator MUST be able to select and download one or more models from
  Hugging Face into the controlled local project model directory.
- **FR-005**: For each configured model, the administrator MUST be able to see its
  identifier, revision when available, status (`not downloaded`, `downloading`, `ready`,
  or `failed`), local model directory when available, and cache location when available.
- **FR-006**: The platform MUST reuse a complete local model without downloading it again.
  A model is `ready` only when all expected model and tokenizer files for its configured
  revision are present and model metadata passes validation. Successful loading or test
  inference is not required to set the ready state. Partial, failed, or invalid downloads
  MUST NOT be reported as ready.
- **FR-007**: The administrator MUST be able to explicitly delete a selected downloaded
  model, clear the Hugging Face cache, and refresh the local model list. Cache cleanup
  MUST handle a missing cache directory.
- **FR-008**: The administrator MUST be able to add, edit, remove, save, and select named
  candidate-label sets. A saved set MUST contain at least two non-empty unique labels.
  The administrator MAY optionally mark one saved set as the default.
- **FR-009**: The platform MUST reject empty labels and duplicate labels. Duplicate
  comparison MUST trim surrounding whitespace and ignore letter case.
- **FR-010**: Candidate-label sets MUST persist in the configured application
  configuration file `configs/labels.yaml` and remain available after an application
  restart.
- **FR-011**: The user MUST be able to enter text, select a downloaded model, select a
  saved candidate-label set, and submit a classification. A configured default set MAY be
  suggested as the initial selection, but the user MUST select the set to use.
- **FR-012**: For every successful classification, the result MUST show the predicted
  class, its confidence, every candidate label with its score ranked in descending order,
  the model used, and a timestamp.
- **FR-013**: The platform MUST reject empty or whitespace-only text and MUST clearly report
  when fewer than two valid labels or no downloaded model is available.
- **FR-014**: Every successful classification MUST be saved as structured JSON in the
  configured MinIO bucket at `predictions/YYYY/MM/DD/{request_id}.json`.
- **FR-015**: The prediction JSON MUST contain `request_id` (unique), `timestamp`,
  `input_text`, `predicted_class`, `confidence`, `candidate_labels`, `ranked_predictions`,
  model name, Hugging Face identifier, and `application_version`; it MUST contain
  `model_revision` with the configured revision string verbatim when configured and MUST
  omit it when no revision is configured. The JSON MAY also contain session ID and latency
  when available.
- **FR-016**: The default MinIO bucket MUST be `zeroshot-learning-output`, configurable
  through an environment variable, and created automatically if missing. The ADR
  documenting this DNS-compatible substitution for `zeroshot_learning_output` MUST be
  recorded before implementation.
- **FR-017**: The platform MUST clearly report model download, model availability,
  candidate-label configuration, and MinIO failures without exposing credentials.
- **FR-018**: If MinIO cannot persist a classification, the platform MUST report the
  failure and MUST NOT present the prediction as a successful persisted result.
- **FR-019**: The project MUST provide documented commands to start the application and
  MUST provide Docker Compose support to start MinIO locally.
- **FR-020**: The README MUST document setup, usage, testing, model caching and deletion,
  MinIO prediction storage, and troubleshooting.
- **FR-021**: Tests MUST include unit coverage for core services, MinIO integration
  coverage, and an end-to-end successful prediction write to MinIO. Tests MUST mock large
  models when model loading is unnecessary.
- **FR-022**: The repository MUST NOT contain committed credentials, usable secrets,
  `.env` files, or model artifacts. `.env.example` MUST contain placeholders only.
- **FR-023**: The repository MUST contain the constitution, this specification, and the
  subsequent implementation plan, data model, and task list before the project is
  considered complete.
- **FR-024**: Each configured model MUST include an optional `revision` field. When
  present, the platform MUST use and display that revision; when absent, the platform
  MUST report that no revision is configured.
- **FR-025**: The platform MUST sanitize submitted text by trimming surrounding whitespace
  and MUST limit it to 10,000 characters after trimming. It MUST reject longer text and
  disallowed control characters with a clear error. Accepted text MUST be treated as plain
  text and safely escaped when displayed.
- **FR-026**: Every prediction MUST have a request ID that is unique across persisted
  prediction objects and is used in its MinIO object path.
- **FR-027**: An automated CI workflow MUST run Ruff lint and format checks, unit tests,
  MinIO integration tests, and the end-to-end prediction-to-MinIO test for each submitted
  change.
- **FR-028**: The application MUST bind to the local loopback interface only. Its
  administrator workflow MUST NOT be reachable from other devices on the network.

### Non-Functional Requirements

- **NFR-001**: Model identifiers, model revisions when available, Python and dependency
  versions, PyTorch and Transformers versions, and MinIO configuration MUST be pinned or
  documented sufficiently for a new developer to reproduce the project locally.
- **NFR-002**: Inference MUST use PyTorch and MUST NOT silently switch to another backend
  or deep-learning framework.
- **NFR-003**: Credentials MUST be supplied through environment variables or secret
  management; secrets MUST NOT be included in logs or committed files. Non-secret model
  and label configuration MUST be read from `configs/models.yaml` and `configs/labels.yaml`.
- **NFR-004**: Structured logs MUST identify model downloads, inference, cache cleanup,
  and MinIO writes without unnecessarily recording sensitive input text.
- **NFR-005**: The application MUST give clear, actionable messages for invalid input,
  unavailable models, failed downloads, invalid candidate-label configuration, and
  MinIO failures.
- **NFR-006**: On every submission, the GitHub Actions workflow MUST run Ruff lint and
  format checks, unit tests, MinIO integration tests, and the end-to-end
  prediction-to-MinIO test. The workflow MUST start MinIO itself and wait for its health
  check before running storage tests, and MUST provide non-secret CI-only MinIO
  credentials through workflow environment variables. The end-to-end test MUST use a
  mocked model pipeline so CI downloads no model artifacts. Final acceptance-criteria
  verification MUST also be performed at completion as specified in T029.
- **NFR-007**: Administrator access MUST be restricted to the same machine by binding to
  loopback only. The workflow has no authentication and MUST NOT be represented as
  production-grade access control.
- **NFR-008**: The user interface, model management, classification, configuration,
  MinIO storage, and logging MUST have separate responsibilities.
- **NFR-009**: The platform MUST reuse a loaded model across consecutive classifications
  rather than repeatedly reloading it, and MUST avoid retaining multiple large models in
  memory when they are not needed.

### Security Considerations

- Credentials MUST NOT appear in logs or Git; `.env.example` MUST contain placeholders
  only.
- Classification input MUST be validated against the size and content rules before
  inference, treated as plain text, and safely escaped when displayed.
- Model and cache deletion MUST require an explicit administrator action and confirmation.
- Administrator functions are available only on the same machine through the loopback
  interface, with no authentication. They MUST NOT be exposed to other devices or
  represented as remotely secured or production-safe controls.

### Acceptance Criteria

1. The application starts with documented commands.
2. MinIO starts locally through Docker Compose.
3. The configured MinIO bucket, defaulting to `zeroshot-learning-output`, is created
   automatically when missing.
4. The administrator sees at least the two required zero-shot models, and additional
   models can be added through configuration only; each model may specify a revision.
5. The administrator can download the DeBERTa model and the second model.
6. Already-downloaded complete models are reused without another download.
7. The administrator can explicitly delete a local model and clear the Hugging Face
   cache, including when the cache directory is missing.
8. A user selects a saved label set and downloaded model, then sees the predicted class,
   confidence, and ranked results.
9. Every successful prediction is saved as JSON in MinIO with the required metadata,
   unique request ID, and date-based object path.
10. Empty, oversized, or invalid text; missing models; invalid labels; and MinIO errors
    produce clear messages.
11. On every submission, CI runs Ruff lint and format checks, unit tests, MinIO integration
    tests, and the mocked-pipeline end-to-end prediction-to-MinIO test after starting and
    health-checking MinIO.
12. The README covers setup, usage, testing, and troubleshooting, as well as caching,
    deletion, and storage.
13. No secrets or model files are committed.
14. The constitution, specification, implementation plan, data model, and task list are
    present in the repository before project completion.
15. Repeated classification with the same model does not repeatedly reload it, and unused
    large models are not retained unnecessarily in memory.
16. The documented administrator workflow allows a default label set to be suggested,
    while each user selects the set used for classification.
17. Text over 10,000 characters or containing disallowed control characters is rejected
    before inference with a clear error.
18. Administrator controls are reachable from the same machine only and cannot be reached
    by another device on the local network.

### Key Entities *(include if feature involves data)*

- **Configured Model**: A selectable zero-shot model identified by its name, Hugging Face
  identifier, configured revision, local directory, cache location, and current lifecycle
  status.
- **Candidate-Label Set**: A named, persisted collection of at least two unique,
  non-empty labels managed by an administrator and selected for classification.
- **Prediction**: A successful classification containing unique request identity,
  timestamp, input text, predicted class, confidence, candidate labels and ranked scores,
  model identity and optional revision, application version, and optional session ID or
  latency.
- **Administrator Action**: An explicit model or candidate-label management operation,
  including download, deletion, cache cleanup, refresh, or label-set changes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The administrator can identify and start downloads for both required models
  from the model-management workflow.
- **SC-002**: A second request to use a complete local model causes zero additional model
  downloads, and consecutive classifications with that model do not reload it.
- **SC-003**: Every successful classification produces exactly one inspectable, durable
  prediction record containing all required metadata; a failed write produces no
  success-shaped confirmation.
- **SC-004**: In validation, all specified administrator and user acceptance scenarios
  pass, including repeated use, explicit deletion, and storage-failure cases.
- **SC-005**: A new developer can follow the README to start the local application and
  storage service and run the test suite without committing credentials or model files.
- **SC-006**: A project maintainer can complete the documented demo end to end: start
  MinIO, download a model, classify text, verify the prediction JSON in MinIO, explicitly
  delete the model or clear the cache, and run the test suite.

## Assumptions

- The v1 deployment is a local learning and demonstration environment; it does not
  provide production identity management or multi-tenant access control.
- Model downloads require network access to Hugging Face and sufficient local disk space.
- Candidate-label uniqueness ignores case and surrounding whitespace; label display
  retains the administrator's entered casing after trimming.
- Timestamps are recorded in ISO 8601 UTC. The date path uses that timestamp's UTC date.
- Confidence and ranked scores use the classifier's normalized score scale from 0 to 1.
- A prediction is successful only after its JSON object has been written to MinIO.
- Candidate-label sets persist in `configs/labels.yaml`; model definitions persist in
  `configs/models.yaml`. Both files contain non-secret project configuration only.
- The maximum submitted text length is 10,000 characters after trimming surrounding
  whitespace; this default is documented for consistent validation.
