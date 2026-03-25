# Code Agent

A production-grade AI software engineering agent powered by **Google ADK 1.27** and **Anthropic Claude** (via LiteLLM). It behaves like a Staff Software Engineer — building projects from scratch, analyzing codebases, writing and reviewing code, debugging issues, running security scans, and managing Jira tickets.

Exposes an **A2A (Agent-to-Agent) protocol** endpoint and an **ADK dev UI**, both served from a single `main.py` using ADK's `get_fast_api_app`.

## Architecture

```
Orchestrator (Alex — Staff SWE)
├── code_navigator   — semantic codebase exploration, symbol lookup, cross-repo search
├── code_writer      — production-quality code, new files, feature additions, refactoring
├── pr_reviewer      — structured PR reviews: correctness, security, performance
├── architect        — new project design, ADRs, data models, API surface
├── debugger         — root cause analysis, stack traces, hypothesis-driven debugging
├── git_agent        — commits, branches, diffs, history, push (with human approval)
├── vcs_agent        — GitHub/Bitbucket API: list repos, PRs, create PRs, remote file access
├── security_agent   — OWASP Top 10, secret detection, CVEs, license compliance
├── docs_agent       — docstrings, READMEs, ADRs, Mermaid diagrams
├── pr_review_pipeline   — end-to-end: fetch diff → navigate → review → security scan
└── feature_pipeline     — end-to-end: understand → implement → docs → commit → PR
```

## Stack

| Layer | Choice |
|---|---|
| Agent framework | Google ADK 1.27.2 |
| LLM | Anthropic Claude (default: `claude-sonnet-4-6`) via LiteLLM |
| Embeddings | Google Gemini (`gemini-embedding-001`) via API key |
| RAG / vector store | LlamaIndex + ChromaDB (local) / pgvector Aurora (EKS) |
| Session storage | ADK InMemory (local) / Aurora PostgreSQL (EKS) |
| VCS | GitHub (PyGithub) + Bitbucket (atlassian-python-api) |
| Project management | Jira + Confluence |
| Runtime | Python 3.12, uv |

## Quick Start

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- Google API key (Gemini embeddings only) — [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Setup

```bash
git clone <repo>
cd code-agent
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and GOOGLE_API_KEY
uv sync
```

### Run

**Production server** (A2A + dev UI, port 8000)
```bash
uv run python main.py
# or via installed script:
uv run code-agent
```

**ADK dev UI only** (development, with file-watch reload)
```bash
uv run adk web .
# With A2A endpoints:
uv run adk web . --a2a
```

**Terminal chat**
```bash
uv run adk run code_agent
```

## Environment Variables

```bash
# LLM — any LiteLLM provider string
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=anthropic/claude-sonnet-4-6   # default; swap to e.g. openai/gpt-4o

# Embeddings (Gemini only — no GCP services needed)
GOOGLE_API_KEY=AIza...

# Deployment
DEPLOYMENT_MODE=local           # local | eks
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=info
API_KEY=                        # optional; enables X-API-Key auth on the custom A2A server

# EKS only
DATABASE_URL=postgresql+asyncpg://user:pass@aurora-host:5432/codeagent

# VCS (optional — required for GitHub/Bitbucket tools)
GITHUB_TOKEN=ghp_...
BITBUCKET_USERNAME=...
BITBUCKET_APP_PASSWORD=...

# Jira / Confluence (optional)
JIRA_URL=https://your-org.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=...
CONFLUENCE_URL=https://your-org.atlassian.net/wiki
```

## A2A Protocol

The agent serves the A2A protocol at `/a2a/code_agent/`.

### Agent card (discovery)
```bash
curl http://localhost:8000/a2a/code_agent/.well-known/agent-card.json
```

### Send a task (ADK `/run`)
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "code_agent",
    "user_id": "user1",
    "session_id": "sess1",
    "new_message": {
      "role": "user",
      "parts": [{"text": "Analyze the project at /path/to/project"}]
    }
  }'
```

### Streaming (SSE)
```bash
curl -N -X POST http://localhost:8000/run_sse \
  -H "Content-Type: application/json" \
  -d '{
    "app_name": "code_agent",
    "user_id": "user1",
    "session_id": "sess1",
    "new_message": {
      "role": "user",
      "parts": [{"text": "Review PR #42 in github.com/myorg/myrepo"}]
    }
  }'
```

### Custom A2A server (JSON-RPC 2.0)

The original A2A server (`code_agent.a2a.server`) is still available via `code-agent-a2a` and supports additional methods: `tasks/submit` (async), `tasks/get`, `tasks/cancel`, `tasks/resume` (human-in-the-loop).

```bash
uv run code-agent-a2a

# Agent card
curl http://localhost:8000/.well-known/agent.json

# Async submit
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "1", "method": "tasks/submit",
    "params": {
      "message": {"parts": [{"type": "text", "text": "Run a security scan on /path/to/project"}]}
    }
  }'
```

## What the Agent Can Do

| Capability | Example prompt |
|---|---|
| Analyze codebase | "Analyze the architecture of /path/to/project" |
| Implement feature | "Implement PROJ-123 — add OAuth2 login" |
| Review PR | "Review PR #42 in github.com/myorg/myrepo" |
| Fix bug | "This error occurs in prod: `<stacktrace>` — find the root cause" |
| Security audit | "Scan the auth module for OWASP Top 10 vulnerabilities" |
| Scaffold project | "Create a FastAPI CRUD app with PostgreSQL in /tmp/myapp" |
| Semantic search | "Find all places where JWT tokens are validated" |
| Write tests | "Write pytest unit tests for the PaymentService class" |
| Git workflow | "Commit staged changes with a Conventional Commits message" |
| Jira integration | "Mark PROJ-456 as In Review and add a comment with the PR URL" |

## Deployment

### Local

```bash
uv run python main.py
```

### Docker

```bash
docker build -t code-agent .
docker run -p 8000:8000 --env-file .env code-agent
```

### Kubernetes (EKS)

```bash
# Set DEPLOYMENT_MODE=eks and DATABASE_URL in your k8s secrets/configmap
kubectl apply -f k8s/
```

In EKS mode the agent uses Aurora PostgreSQL for session persistence. Semantic search uses pgvector (LlamaIndex `PGVectorStore`).

## Guardrails

Five layers of defence applied at different points in the ADK execution pipeline.

| Layer | Where | What it does |
|---|---|---|
| **1 — Provider safety** | `agent.py` `generate_content_config` | Google's model blocks DANGEROUS_CONTENT, HARASSMENT, HATE_SPEECH, SEXUALLY_EXPLICIT before returning. Active when `LLM_MODEL` is a Gemini model; silently skipped for LiteLLM non-Gemini backends. |
| **2a — before_model_callback** | `code_agent/guardrails.py` | Scans user text for injection patterns ("ignore previous instructions", "jailbreak", etc.) → returns a canned `LlmResponse` to skip the model entirely. |
| **2b — after_model_callback** | `code_agent/guardrails.py` | Redacts any API key, token, or credential that accidentally appears in model output (`api_key=`, `sk-…`, `ghp_…`, etc.). |
| **2c — before_tool_callback** | `code_agent/guardrails.py` | Blocks `google_search` calls whose query contains dangerous payloads (`drop table`, `exec(`, `eval(`, etc.). |
| **3 — output_schema** | `agent.py` (commented) | Uncomment `output_schema=AgentResponse` to force structured Pydantic output — this disables tool use entirely. |
| **4 — max_llm_calls** | `code_agent/a2a/callbacks.py` + `RunConfig` | In-callback hard limit of 40 LLM calls per invocation. Also configurable at runtime via `RunConfig(max_llm_calls=N)`. |
| **5 — System instruction** | `agent.py` `_INSTRUCTION` | Hard-coded non-negotiable rules: never reveal system prompt, only retrieve external facts via approved tools, always cite sources. |

### Layer 4 via RunConfig (optional)

```python
from google.adk.runners import RunConfig

await runner.run_async(
    ...,
    run_config=RunConfig(max_llm_calls=20),  # override per invocation
)
```

## Evaluation

Guardrail unit tests run without any API keys or live services.

```bash
# Run all guardrail evals
uv run pytest evals/ -v

# Run a specific layer
uv run pytest evals/ -v -k "Layer2a"
uv run pytest evals/ -v -k "Layer2b"
uv run pytest evals/ -v -k "Layer4"

# Show test IDs only (no output capture)
uv run pytest evals/ --collect-only
```

Test cases are documented in `evals/datasets/guardrail_cases.json`.

## Development

```bash
# Run guardrail evals
uv run pytest evals/ -v

# Health check
curl http://localhost:8000/health

# Change LLM provider without code changes
LLM_MODEL=openai/gpt-4o uv run python main.py
LLM_MODEL=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 uv run python main.py
```
