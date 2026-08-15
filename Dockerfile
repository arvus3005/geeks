FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY src/ src/
COPY configs/ configs/

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "uvicorn", "hhgoa_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
