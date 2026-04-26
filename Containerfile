# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY README.md .

# Install dependencies using uv
RUN uv pip install --system -e ".[dev]"

# Runtime stage
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/

# Set environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default command (can be overridden in compose)
CMD ["uvicorn", "rag_ebook_search.main:app", "--host", "0.0.0.0", "--port", "8000"]
