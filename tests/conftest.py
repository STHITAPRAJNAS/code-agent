"""pytest configuration — stub only the heavy ADK/LiteLLM imports.

code_agent/__init__.py does ``from .agent import root_agent`` which pulls in
google-adk[extensions] (LiteLLM) that is not installed in the test environment.
We stub exactly those two modules so the services/ and tools/ subpackages can
be imported cleanly without the full ADK dependency graph.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# Ensure the project root is on sys.path so ``import code_agent.services.*``
# resolves to the actual source files.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _stub_module(name: str) -> MagicMock:
    """Register a MagicMock under *name* in sys.modules and return it."""
    mock = MagicMock(spec=ModuleType)
    mock.__name__ = name
    mock.__path__ = []   # make it look like a package so sub-imports don't break
    mock.__spec__ = None
    sys.modules[name] = mock
    return mock


# --- Stub the ADK LiteLLM extension (requires pip install google-adk[extensions]) ---
_stub_module("google.adk.models.lite_llm")

# --- Stub code_agent.agent + code_agent.models so __init__.py doesn't blow up ---
# These stubs must be in place BEFORE code_agent/__init__.py is executed.
_agent_stub = _stub_module("code_agent.agent")
_agent_stub.root_agent = MagicMock()

_models_stub = _stub_module("code_agent.models")
_models_stub.default_model = MagicMock(return_value="stub-model")
