# MCP tool server.
#
# The embedding model is downloaded at build time so the container starts
# ready to answer instead of fetching 0.22 GB on its first request.

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first: it only rebuilds when the lockfile changes.
COPY agent/pyproject.toml agent/uv.lock ./
RUN uv sync --frozen --no-install-project

COPY agent/rag ./rag
COPY agent/mcp_server ./mcp_server
RUN uv sync --frozen

# Warm the model cache into the image.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8300

EXPOSE 8300

CMD ["uv", "run", "python", "-m", "mcp_server"]
