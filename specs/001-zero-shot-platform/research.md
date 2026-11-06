# Research: Zero-Shot Classification Platform

## Decisions

### Transformers zero-shot inference

- **Decision**: Use the Transformers `zero-shot-classification` pipeline and explicitly
  select the PyTorch framework. Configure the model and revision from
  `configs/models.yaml`; both shipped model entries use pinned full commit SHAs.
- **Rationale**: The pipeline directly supports NLI-based zero-shot classification and
  candidate-label scoring, matching the required model families and user workflow.
  Configured revisions are passed to model loading and recorded verbatim in prediction
  records. The baseline SHAs are verified against the Hugging Face model pages and never
  use `main`. Additional entries may use a branch, tag, or commit, or omit revision; a
  mutable branch is a documented reproducibility trade-off, while immutable commits are
  preferred.
- **Pinned baselines**: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` uses revision
  `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` and `local_dir`
  `deberta-v3-base-zeroshot-v2.0`. `facebook/bart-large-mnli` uses revision
  `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` and `local_dir` `bart-large-mnli`.
  Revisions were resolved from the [DeBERTa model API](https://huggingface.co/api/models/MoritzLaurer/deberta-v3-base-zeroshot-v2.0)
  and [BART model API](https://huggingface.co/api/models/facebook/bart-large-mnli)
  on 2026-11-05.
- **Alternatives considered**: Load model and tokenizer classes directly for custom
  inference. This adds preprocessing and score-handling code without a current spec need.
- **Sources**: [Transformers pipelines API](https://huggingface.co/docs/transformers/main_classes/pipelines);
  [BART MNLI model card](https://huggingface.co/facebook/bart-large-mnli);
  [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

### Model download, readiness, and cache lifecycle

- **Decision**: Download a configured revision into a unique staging directory on the
  same filesystem as its managed model directory. Validate expected model/tokenizer files
  and metadata, then atomically rename staging into a previously nonexistent
  revision-qualified destination. Reuse an existing complete destination. Remove failed
  staging data, not a valid existing snapshot.
- **Path rule**: Resolve paths as `MODEL_DIR/<local_dir>/<revision or "unpinned">`.
  Use the explicit safe `local_dir` mapping rather than deriving a path from the
  slash-separated Hub model ID. Sanitize non-SHA revision strings for filesystem safety
  while retaining the original revision for inference and prediction records.
- **Rationale**: A same-filesystem rename prevents the application from exposing an
  incomplete download as ready. Keeping immutable revision-qualified directories avoids
  replacing a non-empty active directory, which is not generally portable as an atomic
  operation.
- **Readiness rule**: Follow the clarified spec: expected model and tokenizer files must
  be present and model metadata must validate. Successful load or test inference is not
  required to mark a model ready.
- **Cache cleanup**: Keep application-managed model directories distinct from Hugging Face
  cache paths. Cache cleanup is an explicit action and must tolerate a missing cache
  directory; do not directly manipulate shared Hub cache internals.
- **Alternatives considered**: Download directly into the final directory. This makes
  partial files visible and creates recovery ambiguity. Rely solely on the Hub cache.
  This does not satisfy the requested controlled project model directory and visible
  model deletion workflow.
- **Sources**: [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download);
  [Hub cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache);
  [Python temporary files](https://docs.python.org/3/library/tempfile.html);
  [Python atomic replacement](https://docs.python.org/3/library/os.html#os.replace).

### Streamlit multipage UI and pipeline memory

- **Decision**: Use the requested Streamlit multipage app with UI-only Admin and
  Classify pages. Keep service logic in Python modules. Cache one active model pipeline
  keyed by model directory/revision and inference device; release it before switching to
  another large model.
- **Rationale**: Streamlit's resource cache is intended for heavyweight resources such as
  machine-learning models. A one-active-model policy is consistent with the spec's memory
  constraint and provides predictable model switching.
- **Local-only boundary**: A hidden page is not access control. Run a host app on
  `127.0.0.1`. If containerized, bind inside the container as needed and publish the host
  port only on `127.0.0.1`; do not expose administrator actions to other devices.
- **Alternatives considered**: Retain every model in a multi-key global cache. This can
  retain multiple large models unnecessarily. Add authentication. This is out of v1 scope;
  network binding enforces the explicit same-machine requirement.
- **Sources**: [Streamlit multipage apps](https://docs.streamlit.io/develop/concepts/multipage-apps);
  [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching);
  [`st.cache_resource`](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource);
  [Streamlit server configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml);
  [Compose networking](https://docs.docker.com/compose/how-tos/networking/).

### MinIO prediction persistence and bucket naming

- **Decision**: Use the official MinIO Python client. Ensure the configured bucket exists
  before writes, creating it when missing. Upload a UTF-8 JSON object with a unique
  date-partitioned key and JSON content type. Treat write failure as classification
  failure from the user's perspective.
- **Bucket**: Use `zeroshot-learning-output` as the default. Record an ADR documenting
  that the constitution's original `zeroshot_learning_output` default was substituted
  because S3-compatible bucket naming rules disallow underscores.
- **Rationale**: Idempotent bucket checks satisfy automatic creation; a DNS-compatible
  name avoids a startup failure. Prediction objects remain the durable record.
- **Alternatives considered**: Keep the underscore default and require manual correction
  after a service rejects it. This makes a clean local setup fail unnecessarily.
- **Docker Compose**: Use a persistent MinIO service with health checking and loopback-
  bound host ports. Containers communicate over the Compose network using service names.
  Pin the selected MinIO server image/version and verify its local-use terms during
  implementation; the Python client remains the required storage interface. Treat a
  concurrent bucket-creation "already exists" response as success after rechecking the
  bucket.
- **Sources**: [MinIO Python SDK](https://docs.min.io/aistor/developers/sdk/python/);
  [MinIO Python API](https://docs.min.io/aistor/developers/sdk/python/api/);
  [Compose networking](https://docs.docker.com/compose/how-tos/networking/);
  [MinIO container installation](https://docs.min.io/aistor/installation/container/install/).

### Configuration and secrets

- **Decision**: Read non-secret model and label definitions from `configs/models.yaml`
  and `configs/labels.yaml`. Read MinIO credentials and other deployment-specific secrets
  from environment variables, optionally loaded from a developer `.env` file using
  python-dotenv. Commit `.env.example` with placeholders only.
- **Label-set shape**: Store label entries under `label_sets` and place the optional
  default key once at top level as `default_set`; do not duplicate default state on each
  `LabelSet` entry.
- **Cache variables**: `HF_HOME` is the primary Hub cache variable. Retain
  `TRANSFORMERS_CACHE` for compatibility only, also under the ignored model/cache tree.
- **Git ignore**: Ignore `models/*` and explicitly allow `!models/.gitkeep`.
- **Rationale**: This follows the constitution's explicit allowance for the two
  version-controlled, non-secret YAML files while keeping credentials out of Git.
- **Alternatives considered**: Store model and label definitions only in environment
  variables. This is harder for an administrator to inspect and edit and conflicts with
  the requested YAML configuration workflow.

### Testing and automation

- **Decision**: Invoke tools through Python modules (`python -m pytest`,
  `python -m ruff`) to avoid relying on `.exe` shims that may be blocked on Windows.
  Keep Makefile targets as conveniences but list direct equivalent commands. Mock
  downloads, large model pipelines, and MinIO in unit tests. Run lint and unit tests in
  GitHub Actions. Run MinIO integration tests against local Compose MinIO, and keep the
  full end-to-end demo available as a documented validation sequence.
- **Test locations**: Unit suites are `tests/unit/test_model_manager.py`,
  `tests/unit/test_classifier.py`, and `tests/unit/test_minio_storage.py`.
  `tests/integration/test_minio_storage.py` covers the live MinIO contract.
- **Rationale**: Fast unit tests avoid large downloads and external-service flakiness;
  service integration and end-to-end tests still verify the MinIO object contract.
- **Alternatives considered**: Download and initialize production-sized models in CI.
  This increases runtime and network dependence and is not needed to validate most
  service behavior.
- **Sources**: [pytest documentation](https://docs.pytest.org/);
  [GitHub Actions Python testing](https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python).

## Project Decisions Requiring Follow-up

- Create the bucket-name ADR before implementation, stating the reason, benefits, risks,
  and reproducibility impact of using `zeroshot-learning-output`.
- Pin exact Python, PyTorch, Transformers, Streamlit, MinIO SDK, and other dependency
  versions in project metadata/lockfile. Select and pin the MinIO server image/version.
- If a baseline model is updated to another commit, change its `revision` in
  `configs/models.yaml` and update the documented path and prediction expectations
  together. Prediction records preserve the exact configured revision string.
