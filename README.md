# Code Agent

A production-grade AI software engineering agent powered by **Google ADK** and **Gemini 2.0 Flash**. It behaves like a Staff Software Engineer — building projects from scratch, analyzing codebases, writing code, reviewing PRs, debugging issues, and running shell commands.

Exposes an **A2A (Agent-to-Agent) compatible** API via FastAPI, making it remotely hostable and interoperable with other A2A agents.

## Architecture

```
Orchestrator (Alex — Staff SWE)
├── File Agent        — read/write/search files
├── Shell Agent       — run commands safely
├── Git Agent         — git operations, commits, diffs
├── Code Review Agent — PR review, security, quality
├── Debug Agent       — root cause analysis, bug fixes
└── Architect Agent   — system design, project scaffolding
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google AI Studio API key → [Get one here](https://aistudio.google.com/app/apikey)

### 2. Setup

```bash
git clone <repo>
cd code-agent
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
uv sync
```

### 3. Run

**Option A — FastAPI A2A server (port 8000)**
```bash
uv run code-agent
```

**Option B — ADK interactive web UI (port 8080)**
```bash
uv run adk web code_agent --port 8080
```

**Option C — ADK terminal chat**
```bash
uv run adk run code_agent
```

## A2A API

### Agent Card
```bash
curl http://localhost:8000/.well-known/agent.json
```

### Send a task
```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tasks/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Analyze the project at /path/to/project and give me an overview"}]
      }
    }
  }'
```

### Streaming (SSE)
```bash
curl -N -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tasks/sendSubscribe",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Review this PR diff: <diff here>"}]
      }
    }
  }'
```

## What the Agent Can Do

| Capability | Example prompt |
|------------|---------------|
| Build a project | "Create a FastAPI CRUD app with SQLite in /tmp/myapp" |
| Analyze code | "Analyze the codebase at /path/to/project and summarize the architecture" |
| Modify code | "Add JWT authentication to the FastAPI app at /tmp/myapp" |
| Review PR | "Review this git diff and flag any issues: `<diff>`" |
| Debug | "This error occurs: `<stacktrace>` — find the root cause and fix it" |
| Git ops | "Show me the last 10 commits in /path/to/repo" |
| Run commands | "Run the test suite for the project at /tmp/myapp" |

## Development

```bash
# Run tests
uv run pytest

# Check health
curl http://localhost:8000/health
```
