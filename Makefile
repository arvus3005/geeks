.PHONY: install fmt fmt-check lint typecheck test test-unit ci dry-run-index dry-run-ingest validate-manifest estimate-capacity scan-secrets

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
	uv run pytest tests/ -q -rs

ci: typecheck lint fmt-check test

# Offline dry-runs — require no credentials, make no Pinecone calls
dry-run-index:
	uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi

dry-run-ingest:
	uv run python scripts/ingest_all.py --mode pilot

dry-run-ingest-prepared:
	@echo "Usage: make dry-run-ingest-prepared MANIFEST=artifacts/prepared/<id>_manifest.json"
	@test -n "$(MANIFEST)" || (echo "ERROR: MANIFEST variable not set" && exit 1)
	uv run python scripts/ingest_prepared.py --manifest $(MANIFEST) --dry-run

# Secret scan (filenames only — never prints secret values)
scan-secrets:
	@echo "Scanning for credential literals..."
	@rg -l "pc-[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}" src tests scripts bench || true
	@echo "Scan complete."

# Validate a prepared-canary manifest (offline, no credentials)
validate-manifest:
	@echo "Usage: make validate-manifest MANIFEST=artifacts/prepared/<id>_manifest.json"
	@test -n "$(MANIFEST)" || (echo "ERROR: MANIFEST variable not set" && exit 1)
	uv run python scripts/ingest_prepared.py --manifest $(MANIFEST) --dry-run

# Dry-run the canary indexer (no credentials needed)
# Usage: make canary-dry-run MANIFEST=artifacts/prepared/<id>_manifest.json
canary-dry-run:
	@echo "Usage: make canary-dry-run MANIFEST=artifacts/prepared/<id>_manifest.json"
	@test -n "$(MANIFEST)" || (echo "ERROR: MANIFEST variable not set" && exit 1)
	uv run python scripts/index_canary.py --manifest $(MANIFEST)

# Live canary execution (requires PINECONE_API_KEY and CONFIRM_PINECONE_WRITE=1)
# Usage: CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> make canary-execute MANIFEST=artifacts/prepared/<id>_manifest.json
canary-execute:
	@echo "Usage: CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> make canary-execute MANIFEST=..."
	@test -n "$(MANIFEST)" || (echo "ERROR: MANIFEST variable not set" && exit 1)
	uv run python scripts/index_canary.py --manifest $(MANIFEST) --execute --resume --concurrency 4

# Estimate indexing capacity for the current budget configuration
estimate-capacity:
	uv run python -c "\
from hhgoa_rag.ingestion.budget import make_default_guard; \
g = make_default_guard(); \
r = g.usage_report(); \
print('Budget estimate:'); \
[print(f'  {k}: {v}') for k,v in r.items() if k != 'per_language']; \
"
