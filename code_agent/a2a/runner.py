"""ADK Runner wrapper — bridges the ADK execution model with the A2A task model."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from code_agent.a2a.models import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent

_APP_NAME = "code_agent"


class AgentRunner:
    """Wraps ADK Runner + InMemorySessionService to execute the agent and manage A2A tasks."""

    def __init__(self, agent: "BaseAgent") -> None:
        self._agent = agent
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=agent,
            app_name=_APP_NAME,
            session_service=self._session_service,
        )
        # In-memory task store: task_id -> Task
        self._tasks: dict[str, Task] = {}
        self._canceled: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Mark a task as canceled. Returns True if the task existed."""
        if task_id not in self._tasks:
            return False
        self._canceled.add(task_id)
        task = self._tasks[task_id]
        if task.status.state in (TaskState.SUBMITTED, TaskState.WORKING):
            task.status = TaskStatus(state=TaskState.CANCELED)
        return True

    async def invoke(self, user_message: str, session_id: str | None = None) -> Task:
        """Run the agent synchronously and return a completed Task."""
        task_id = str(uuid4())
        sid = session_id or str(uuid4())

        # Create task record
        user_msg = Message.user(user_message)
        task = Task(
            id=task_id,
            session_id=sid,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[user_msg],
        )
        self._tasks[task_id] = task

        try:
            task.status = TaskStatus(state=TaskState.WORKING)
            response_text = await self._run_adk(user_message, sid)

            if task_id in self._canceled:
                return task  # already marked canceled

            agent_msg = Message.agent(response_text)
            task.history.append(agent_msg)
            task.artifacts = [
                Artifact(
                    index=0,
                    name="response",
                    parts=[TextPart(text=response_text)],
                )
            ]
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=agent_msg,
            )
        except Exception as exc:
            error_msg = Message.agent(f"Error: {exc}")
            task.history.append(error_msg)
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=error_msg,
            )

        return task

    async def stream(
        self, user_message: str, session_id: str | None = None
    ) -> AsyncGenerator[Task, None]:
        """Stream task state updates as the agent runs. Yields Task snapshots."""
        task_id = str(uuid4())
        sid = session_id or str(uuid4())

        user_msg = Message.user(user_message)
        task = Task(
            id=task_id,
            session_id=sid,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[user_msg],
        )
        self._tasks[task_id] = task
        yield task  # initial: submitted

        task.status = TaskStatus(state=TaskState.WORKING)
        yield task  # working

        try:
            response_text = await self._run_adk(user_message, sid)

            if task_id in self._canceled:
                yield task
                return

            agent_msg = Message.agent(response_text)
            task.history.append(agent_msg)
            task.artifacts = [
                Artifact(
                    index=0,
                    name="response",
                    parts=[TextPart(text=response_text)],
                )
            ]
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=agent_msg,
            )
            yield task  # completed

        except Exception as exc:
            error_msg = Message.agent(f"Error: {exc}")
            task.history.append(error_msg)
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=error_msg,
            )
            yield task  # failed

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_adk(self, user_message: str, session_id: str) -> str:
        """Execute one turn through the ADK runner and return the final text response."""
        # Ensure session exists
        existing = await self._session_service.get_session(
            app_name=_APP_NAME, user_id="default", session_id=session_id
        )
        if existing is None:
            await self._session_service.create_session(
                app_name=_APP_NAME, user_id="default", session_id=session_id
            )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        )

        final_text_parts: list[str] = []

        async for event in self._runner.run_async(
            user_id="default",
            session_id=session_id,
            new_message=content,
        ):
            # Collect final response text from the agent's last turn
            if event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_text_parts.append(part.text)

        return "\n".join(final_text_parts) if final_text_parts else "(no response)"
