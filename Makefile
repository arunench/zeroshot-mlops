.PHONY: app lint format-check test test-integration minio-up minio-down

app:
	python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1

lint:
	python -m ruff check .

format-check:
	python -m ruff format --check .

test:
	python -m pytest tests/unit

test-integration:
	python -m pytest tests/integration

minio-up:
	docker compose up -d minio

minio-down:
	docker compose down
