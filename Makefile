.PHONY: install fmt fmt-check lint typecheck test test-unit test-behav ci smoke-up smoke-down

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

ci: typecheck lint fmt-check test-unit

smoke-up:
	docker compose up -d qdrant

smoke-down:
	docker compose down
