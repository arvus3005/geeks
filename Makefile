.PHONY: install fmt fmt-check lint typecheck test test-unit ci dry-run-index dry-run-ingest

install:
	uv sync --all-extras

fmt:
	uv run ruff format src/ tests/ bench/ scripts/

fmt-check:
	uv run ruff format --check src/ tests/ bench/ scripts/

lint:
	uv run ruff check src/ tests/ bench/ scripts/

typecheck:
	uv run mypy src/

test-unit:
	uv run pytest tests/unit/ tests/behavioural/ tests/contract/ -v

test:
	uv run pytest tests/ -v --ignore=tests/integration

ci: typecheck lint fmt-check test

# Offline dry-runs — require no credentials, make no Pinecone calls
dry-run-index:
	uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi

dry-run-ingest:
	uv run python scripts/ingest_all.py --mode pilot

# Secret scan (filenames only — never prints secret values)
scan-secrets:
	@echo "Scanning for credential literals..."
	@rg -l "pc-[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}" src tests scripts bench || true
	@echo "Scan complete."
