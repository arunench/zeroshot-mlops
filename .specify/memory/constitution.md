<!--
Sync Impact Report
Version change: 1.0.0 to 1.1.0 (MINOR; expanded scope, stack, and persistence requirements)
Modified principles: Purpose and Scope; Reproducible, Fixed Technology Stack; Persistence and Component Boundaries
Added sections: none
Removed sections: none
Follow-up TODOs: none
-->
# Zero-Shot Text Classification MLOps Platform Constitution

## Core Principles

### I. Purpose and Scope
This project is a learning and demonstration platform for MLOps practices around zero-shot
text classification. Version 1 MUST support at least two zero-shot models, including
`MoritzLaurer/deberta-v3-base-zeroshot-v2.0` and `facebook/bart-large-mnli`. Additional
models MUST be addable through configuration only, without application code changes. V1
MUST provide a distinct administrator workflow for model and candidate-label management
and a separate user workflow for classification. It MUST provide a reproducible local
workflow to configure the platform, classify text through the Streamlit interface, manage
Hugging Face models, and persist successful predictions in MinIO. V1 is not a model-training
or fine-tuning system, general-purpose model-serving framework, distributed orchestration
platform, or production multi-tenant service. Scope boundaries keep the implementation
teachable and verifiable.

### II. Reproducible, Fixed Technology Stack
The mandatory stack is Python, PyTorch, Hugging Face Transformers, Streamlit, and MinIO.
Python MUST always be the implementation language. Other languages MAY be used only for
supporting configuration, infrastructure, or deployment files, such as YAML, JSON,
Dockerfiles, or shell scripts. Inference MUST use the PyTorch backend and MUST NOT silently
fall back to another backend. No other deep-learning framework may be introduced without a
formal ADR. Python, dependencies, PyTorch and Transformers versions, Hugging Face model
identifiers and revisions, and MinIO configuration MUST be pinned or explicitly documented
so another developer can reproduce the same setup. Changing any part of the mandatory
stack requires a documented ADR stating the reason, benefits, risks, and impact on
reproducibility.

### III. Controlled Model Lifecycle
Models MUST be downloaded from Hugging Face at a documented revision and stored in a
controlled local directory. The model manager MUST reuse a complete, validated local model
when available and MUST NOT mark or use partial downloads as ready. Cached model artifacts
MUST remain in place until an explicit administrator action requests their removal.
Download, readiness, reuse, and cleanup behavior MUST be observable and documented.

### IV. Persistence and Component Boundaries
Every successful classification MUST be saved as structured JSON in the configured MinIO
bucket; MinIO is the system of record for predictions. The default bucket MUST be
`zeroshot_learning_output` and MUST be configurable through an environment variable. The
bucket MUST be created automatically if it does not exist. If MinIO's bucket-naming rules
reject that default, a documented ADR MAY substitute a compliant name, such as
`zeroshot-learning-output`. Local files MAY be used only for temporary processing, model
caching, or tests, not as an alternate prediction record.
The Streamlit UI, model manager, classifier, configuration, MinIO storage, and logging
MUST be separate components with clear responsibilities. This separation makes the
platform testable and keeps user-interface behavior independent of core services.

### V. Verification, Observability, and Privacy
Core services MUST have unit tests; MinIO behavior MUST have integration tests; tests MUST
mock large models where loading them is unnecessary. Structured logs MUST cover model
downloads, inference, cache cleanup, and MinIO writes. Logs and errors MUST avoid
unnecessary sensitive input and MUST never reveal credentials. Invalid input, unavailable
models, failed downloads, and MinIO failures MUST produce clear, actionable messages.
These requirements make failures diagnosable without compromising user data.

## Platform Constraints and Operations

Configuration and credentials MUST come only from environment variables or a secret
management mechanism. Secrets and credentials MUST NOT be committed. `.env.example` MUST
contain placeholders only, never usable credentials.

MinIO endpoint, bucket, and access configuration MUST be documented and reproducibly
configured. MinIO MUST be available for integration and end-to-end testing. Local
administrator access to cached models and cleanup operations is a documented trust
boundary, not a substitute for production access controls.

Project documentation MUST cover setup, usage, model caching and explicit deletion, MinIO
prediction storage, testing, and troubleshooting. The README MUST enable a new developer
to reproduce the project locally.

## Development Workflow and Compliance

The test suite MUST run on every submission. Linting, formatting, automated tests, and
documented acceptance criteria MUST pass before a change is considered complete. At least
one end-to-end test MUST perform a successful classification and write its prediction to
MinIO. Reviews MUST verify that no model artifacts, model caches, `.env` files, or
credentials are tracked in Git.

Compliance is checked by the automated test suite and quality checks, inspection of Git
contents for forbidden artifacts and secrets, verification of the MinIO end-to-end test,
and confirmation that the README supports a local reproduction. Any deliberate exception
to a principle requires a documented ADR stating the affected principle, reason, benefits,
risks, and impact on reproducibility. An ADR records an exception but does not amend this
constitution; amendments follow Governance.

## Governance

This constitution governs project decisions and supersedes conflicting local conventions.
An amendment MUST document the motivation, affected principles, compliance impact, and any
necessary migration or follow-up work. Changes MUST be reviewed against existing code,
tests, documentation, and artifacts for compliance. Exceptions require a documented ADR;
they do not waive the amendment process when a principle itself must change.

Constitution versions use semantic versioning. A MAJOR increment is required for
backward-incompatible governance changes, including removing or redefining a principle.
A MINOR increment is required for a new principle or materially expanded guidance. A
PATCH increment is for clarifications and non-semantic refinements. Every amendment MUST
update the version and last-amended date and include a Sync Impact Report for review.

The test suite and quality gates MUST run on every submission. Reviewers MUST confirm
stack and configuration reproducibility, absence of committed model artifacts/caches and
secrets, the MinIO end-to-end persistence test, and README reproduction instructions.

**Version**: 1.1.0 | **Ratified**: 2026-11-05 | **Last Amended**: 2026-11-05
