FROM python:3.12-slim

# System deps: git, ripgrep
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ripgrep curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY code_agent/ ./code_agent/

ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/tmp/code_agent_workspaces
ENV CHROMA_PATH=/data/chroma_db

EXPOSE 8000

VOLUME ["/data/chroma_db", "/tmp/code_agent_workspaces"]

CMD ["uv", "run", "code-agent"]
