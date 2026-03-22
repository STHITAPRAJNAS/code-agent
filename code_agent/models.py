"""Shared LLM model factory.

Uses Google ADK's LiteLlm wrapper so any LiteLLM-supported provider
can be configured via the LLM_MODEL environment variable.

The value should be a full LiteLLM model string, e.g.:
  anthropic/claude-sonnet-4-6   ← default (requires ANTHROPIC_API_KEY)
  openai/gpt-4o                 (requires OPENAI_API_KEY)
  bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0  (requires AWS creds)

All agents call default_model() — changing LLM_MODEL at runtime
switches every agent to the new provider with no code changes.
"""

import os
from google.adk.models.lite_llm import LiteLlm


def default_model() -> LiteLlm:
    """Return a LiteLlm instance configured from LLM_MODEL env var."""
    model_id = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-6")
    return LiteLlm(model=model_id)
