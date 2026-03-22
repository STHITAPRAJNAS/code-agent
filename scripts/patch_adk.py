#!/usr/bin/env python3
"""Apply upstream ADK patches to the installed google-adk package.

Run this after any ``pip install --upgrade google-adk`` to re-apply patches
that are pending upstream merge:

  uv run python scripts/patch_adk.py

Patches applied
---------------
1. agent_to_a2a.py  — adds ``task_store: Optional[TaskStore] = None`` to
   ``to_a2a()``.  Upstream: https://github.com/google/adk-python/pull/3839

2. fast_api.py      — adds ``a2a_task_store`` and ``a2a_push_config_store``
   params to ``get_fast_api_app()``.  Follows the same pattern as PR #3839
   extended to the higher-level function.

Remove this script and the corresponding code in ``main.py`` / ``stores.py``
once google-adk >= the version that merges these features is released.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_package_file(package_path: str) -> Path:
    """Locate an installed package file by its import-style path."""
    spec = importlib.util.find_spec(package_path.split(".")[0])
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"Package not found: {package_path}")
    root = Path(list(spec.submodule_search_locations)[0]).parent
    relative = Path(*package_path.split(".")).with_suffix(".py")
    target = root / relative
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    return target


def _apply(path: Path, old: str, new: str, description: str) -> bool:
    """Replace `old` with `new` in `path`. Returns True if changed."""
    source = path.read_text(encoding="utf-8")
    if new in source:
        print(f"  [skip] already patched: {description}")
        return False
    if old not in source:
        print(f"  [WARN] patch anchor not found — ADK may have changed: {description}")
        return False
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    print(f"  [ok]   {description}")
    return True


# ── Patch 1: agent_to_a2a.py — add TaskStore import ─────────────────────────

AGENT_TO_A2A_PKG = "google.adk.a2a.utils.agent_to_a2a"

PATCH_1A_OLD = """\
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import PushNotificationConfigStore
from a2a.types import AgentCard"""

PATCH_1A_NEW = """\
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import PushNotificationConfigStore
from a2a.server.tasks import TaskStore
from a2a.types import AgentCard"""

PATCH_1B_OLD = """\
def to_a2a(
    agent: BaseAgent,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: Optional[Union[AgentCard, str]] = None,
    push_config_store: Optional[PushNotificationConfigStore] = None,
    runner: Optional[Runner] = None,
) -> Starlette:"""

PATCH_1B_NEW = """\
def to_a2a(
    agent: BaseAgent,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: Optional[Union[AgentCard, str]] = None,
    task_store: Optional[TaskStore] = None,
    push_config_store: Optional[PushNotificationConfigStore] = None,
    runner: Optional[Runner] = None,
) -> Starlette:"""

PATCH_1C_OLD = "  task_store = InMemoryTaskStore()"
PATCH_1C_NEW = "  task_store = task_store or InMemoryTaskStore()"

# ── Patch 2: fast_api.py — add a2a_task_store / a2a_push_config_store ────────

FAST_API_PKG = "google.adk.cli.fast_api"

PATCH_2A_OLD = """\
    web: bool,
    a2a: bool = False,
    host: str = "127.0.0.1","""

PATCH_2A_NEW = """\
    web: bool,
    a2a: bool = False,
    a2a_task_store: Optional[Any] = None,
    a2a_push_config_store: Optional[Any] = None,
    host: str = "127.0.0.1","""

PATCH_2B_OLD = "      a2a_task_store = InMemoryTaskStore()"
PATCH_2B_NEW = "      a2a_task_store = a2a_task_store or InMemoryTaskStore()"

PATCH_2C_OLD = "          push_config_store = InMemoryPushNotificationConfigStore()"
PATCH_2C_NEW = "          push_config_store = a2a_push_config_store or InMemoryPushNotificationConfigStore()"


def main() -> None:
    changed = 0

    print(f"Patching agent_to_a2a.py  ({AGENT_TO_A2A_PKG})")
    try:
        f = _find_package_file(AGENT_TO_A2A_PKG)
        changed += _apply(f, PATCH_1A_OLD, PATCH_1A_NEW, "add TaskStore import")
        changed += _apply(f, PATCH_1B_OLD, PATCH_1B_NEW, "add task_store param to to_a2a()")
        changed += _apply(f, PATCH_1C_OLD, PATCH_1C_NEW, "use task_store or InMemoryTaskStore()")
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        sys.exit(1)

    print(f"\nPatching fast_api.py  ({FAST_API_PKG})")
    try:
        f = _find_package_file(FAST_API_PKG)
        changed += _apply(f, PATCH_2A_OLD, PATCH_2A_NEW, "add a2a_task_store + a2a_push_config_store params")
        changed += _apply(f, PATCH_2B_OLD, PATCH_2B_NEW, "use a2a_task_store or InMemoryTaskStore()")
        changed += _apply(f, PATCH_2C_OLD, PATCH_2C_NEW, "use a2a_push_config_store or InMemory...()")
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        sys.exit(1)

    print(f"\nDone — {changed} patch(es) applied.")


if __name__ == "__main__":
    main()
