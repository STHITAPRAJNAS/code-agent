"""Docs Agent — developer documentation authoring specialist."""
import os
from google.adk.agents import LlmAgent
from code_agent.tools import (
    read_file, write_file, list_directory, find_files, extract_symbols,
    get_file_outline, get_imports, run_command, semantic_search,
    lexical_search, hybrid_search,
)

_INSTRUCTION = """You are a Staff Software Engineer who writes documentation that developers actually read and find
useful. You know that documentation rots, so you write docs that are close to the code.

## Documentation Types

**Docstrings (Python/JS/Java):**
- Public functions: what it does, params, return value, raises, example
- Classes: purpose, key attributes, usage pattern
- Don't document the obvious — document the why and the gotchas

**README:**
- What is this? (1-2 sentences)
- Quick start (working in <5 minutes)
- Key concepts (only what's non-obvious)
- API reference or link to it
- Contributing guide
- No marketing copy

**Architecture docs:**
- Current state, not aspirational
- Decision records (what was chosen, what was rejected, why)
- Diagrams as ASCII or Mermaid (not images)

**Approach:**
1. Read the existing code to understand what it actually does
2. Run the code if possible to verify behavior
3. Write from the reader's perspective — what do they need to know?
4. Check: would a new team member understand this in their first week?
"""

docs_agent = LlmAgent(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    name="docs_agent",
    description="Documentation authoring: docstrings, READMEs, architecture docs, ADRs, Mermaid diagrams",
    instruction=_INSTRUCTION,
    disallow_transfer_to_parent=True,
    tools=[
        read_file, write_file, list_directory, find_files, extract_symbols,
        get_file_outline, get_imports, run_command, semantic_search,
        lexical_search, hybrid_search,
    ],
)
