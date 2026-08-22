FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/

ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/.cache/huggingface

# Bake the local query-embedding model (e5-small ONNX int8, ~118MB) into the
# image at build time so the first real request doesn't pay a HuggingFace
# download — the model loads once from local disk at process startup
# instead (see CLAUDE.md).
RUN uv run python -c "from hhgoa_rag.retrieval.local_embedder import _lazy_load; _lazy_load()"

# Render (and most PaaS Docker runners) inject $PORT and expect the
# container to bind to it; 8000 is the local-dev fallback.
CMD ["sh", "-c", "uv run uvicorn hhgoa_rag.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
