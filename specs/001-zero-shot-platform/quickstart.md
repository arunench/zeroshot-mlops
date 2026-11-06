# Quickstart: Zero-Shot Classification Platform

This guide is the end-to-end acceptance run for the planned local platform. Commands
become executable when implementation artifacts in [plan.md](./plan.md) are delivered.
Run from the repository root.

## Prerequisites

- Python 3.11 and Git
- Docker Engine/Desktop with Docker Compose
- Network access to Hugging Face for the first model download
- Sufficient disk and memory for one configured model

## Setup and start MinIO

1. Copy `.env.example` to `.env`; keep the example placeholders unchanged in Git and put
   only local credentials in `.env`.
2. Set `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` in `.env`. For an app running directly
   on the host, set `MINIO_ENDPOINT=localhost:9000`. For an app running inside Docker
   Compose, set `MINIO_ENDPOINT=minio:9000`.
3. Start MinIO:

   ```powershell
   docker compose up -d minio
   ```

4. Confirm the MinIO health check is passing. The configured
   `MINIO_BUCKET` defaults to `zeroshot-learning-output` and is created by the application
   when missing. Compose host ports for MinIO and the app are bound to loopback; do not
   change them to all-interface bindings for the unauthenticated local administrator UI.

## Install and start the application

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

For a host-run app, `.streamlit/config.toml` also sets `server.address = "127.0.0.1"`;
the command explicitly binds to that address. The host-run app reads
`MINIO_ENDPOINT=localhost:9000`. For an app running inside Compose, set
`MINIO_ENDPOINT=minio:9000` so it uses Compose service discovery and the internal MinIO
port; do not use `localhost` between containers. In Compose, Streamlit listens on
`0.0.0.0` inside its container, and `docker-compose.yml` publishes it only as
`127.0.0.1:8501:8501`. Verify the app cannot be reached from another device on the local
network; do not change the host binding to `0.0.0.0`.

Makefile targets are optional because `make` may be unavailable on Windows. Use the
direct module commands as the portable alternative; no `streamlit.exe`, `ruff.exe`, or
`pytest.exe` shims are required.

| Action | Makefile target | Direct command |
|---|---|---|
| Start app on host | `make app` | `python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1` |
| Lint | `make lint` | `python -m ruff check .` |
| Format check | `make format-check` | `python -m ruff format --check .` |
| Unit tests | `make test` | `python -m pytest tests/unit` |
| MinIO integration tests | `make test-integration` | `python -m pytest tests/integration` |
| Start MinIO | `make minio-up` | `docker compose up -d minio` |
| Stop MinIO | `make minio-down` | `docker compose down` |

## Validate the end-to-end workflow

1. Open the local Streamlit app and select the Admin page.
2. Confirm both required model IDs and pinned revisions are listed in
   `configs/models.yaml`:
   - `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` at
     `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`, mapped to
     `deberta-v3-base-zeroshot-v2.0`.
   - `facebook/bart-large-mnli` at
     `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`, mapped to `bart-large-mnli`.
   Download one model and wait for
   the complete-files and metadata checks to mark it `ready`.
3. Confirm a repeat download reuses the local model. Optionally select the second model.
4. Create and save a candidate-label set with at least two unique labels under
   `label_sets` in `configs/labels.yaml`. To set a suggested default, use the optional
   top-level `default_set: <label-set-key>`; the user must still select a label set.
5. Open Classify Text, explicitly select a ready model and saved label set, enter valid
   text, and submit.
6. Confirm the prediction shows the class, confidence, ranked labels, model, and
   timestamp. Confirm MinIO contains exactly one object at
   `predictions/YYYY/MM/DD/{request_id}.json` with the required fields. The configured
   revision is present verbatim when set and omitted when unset.
7. Test empty, oversized, and invalid-control-character inputs; each must show an error
   and create no prediction object.
8. Use the Admin page to explicitly confirm model deletion or cache cleanup. Confirm the
   model is no longer ready; missing cache directories are handled without an error.
9. Run validation with Python module invocations:

   ```powershell
   python -m ruff check .
   python -m ruff format --check .
   python -m pytest tests/unit/test_model_manager.py tests/unit/test_classifier.py tests/unit/test_minio_storage.py
   ```

10. With MinIO running, run the MinIO integration tests and confirm bucket creation and
    prediction persistence:

    ```powershell
    python -m pytest tests/integration/test_minio_storage.py
    ```

    The end-to-end demo is complete when setup, download,
    classification, JSON inspection, cleanup, and test steps all pass.

## Expected results and failure checks

- MinIO write failure is visible to the user and is not reported as a successful saved
  prediction.
- No credentials, `.env`, downloaded model files, or model cache directories are tracked
  by Git.
- `.gitignore` must ignore `models/*` while allowing `!models/.gitkeep`.
- `HF_HOME` is the primary Hugging Face cache setting. `TRANSFORMERS_CACHE` is retained
  for compatibility, not as the primary cache variable.
- The app is accessible only on the same machine. Do not bind/publish it to a network
  interface accessible to other devices.
- Detailed prediction fields and constraints are defined in
  [data-model.md](./data-model.md) and [prediction.schema.json](./contracts/prediction.schema.json).
