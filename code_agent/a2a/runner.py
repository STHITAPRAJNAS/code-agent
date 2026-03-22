"""ADK Runner wrapper — bridges ADK execution with the A2A task model.

Pause / resume / human-in-the-loop
────────────────────────────────────
When the agent calls a LongRunningFunctionTool (e.g. request_pr_approval):

  1. ADK sets event.actions.long_running_tool_ids on the emitted event and
     run_async() ends — the agent is "paused".
  2. This runner detects the long-running event, extracts the invocation_id
     and function_call_ids, stores them in task.metadata["pending_approval"],
     and sets task.status = INPUT_REQUIRED.
  3. The A2A client sees INPUT_REQUIRED and surfaces the approval details to
     a human operator.
  4. The human calls tasks/resume with their decision.
  5. runner.resume() calls run_async() with invocation_id= and a FunctionResponse
     containing the human's payload, continuing the same invocation.
  6. The agent reads the FunctionResponse, decides to proceed or abort, and
     finishes the turn normally.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from google.adk.runners import Runner
from google.genai import types as genai_types

from code_agent.a2a.models import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from code_agent.a2a.callbacks import (
    before_agent_callback,
    after_agent_callback,
    before_model_callback,
    after_model_callback,
    before_tool_callback,
    after_tool_callback,
)
from code_agent.storage.session_factory import create_session_service

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent

logger = logging.getLogger(__name__)

_APP_NAME = "code_agent"


def _ensure_session_id(session_id: str | None) -> str:
    if session_id and session_id.strip():
        return session_id
    return str(uuid4())


def _extract_long_running_info(event: Any) -> tuple[str | None, list[str]]:
    """Return (invocation_id, [function_call_ids]) if this is a long-running event."""
    try:
        actions = getattr(event, "actions", None)
        if actions is None:
            return None, []
        ids = getattr(actions, "long_running_tool_ids", None) or []
        if not ids:
            return None, []
        invocation_id = getattr(event, "invocation_id", None)
        return invocation_id, list(ids)
    except Exception:
        return None, []


def _extract_tool_name_from_event(event: Any, function_call_id: str) -> str:
    """Try to find the tool name for a given function_call_id in the event content."""
    try:
        if not event.content or not event.content.parts:
            return "unknown"
        for part in event.content.parts:
            fr = getattr(part, "function_response", None)
            if fr and getattr(fr, "id", None) == function_call_id:
                return getattr(fr, "name", "unknown")
    except Exception:
        pass
    return "unknown"


def _extract_tool_response_payload(event: Any) -> dict:
    """Extract the response payload from a LongRunningFunctionTool event."""
    try:
        if not event.content or not event.content.parts:
            return {}
        for part in event.content.parts:
            fr = getattr(part, "function_response", None)
            if fr:
                resp = getattr(fr, "response", None)
                if isinstance(resp, dict):
                    return resp
    except Exception:
        pass
    return {}


class AgentRunner:
    """Wraps ADK Runner + SessionService to execute the agent and manage A2A tasks.

    Shared state
    ────────────
    _tasks         — task_id → Task
    _canceled      — set of task_ids that have been canceled
    _running_tasks — task_id → asyncio.Task (for cancellation)
    """

    def __init__(self, agent: "BaseAgent") -> None:
        self._agent = agent
        self._session_service = create_session_service()
        self._runner = Runner(
            agent=agent,
            session_service=self._session_service,
            app_name=_APP_NAME,
        )

        # All six ADK callbacks
        agent.before_agent_callback = before_agent_callback
        agent.after_agent_callback = after_agent_callback
        agent.before_model_callback = before_model_callback
        agent.after_model_callback = after_model_callback
        agent.before_tool_callback = before_tool_callback
        agent.after_tool_callback = after_tool_callback

        self._tasks: dict[str, Task] = {}
        self._canceled: set[str] = set()
        self._running_tasks: dict[str, asyncio.Task] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        if task_id not in self._tasks:
            return False
        self._canceled.add(task_id)
        task = self._tasks[task_id]
        if task.status.state in (TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED):
            task.status = TaskStatus(state=TaskState.CANCELED)
        running = self._running_tasks.get(task_id)
        if running and not running.done():
            running.cancel()
        return True

    async def invoke(self, user_message: str, session_id: str | None = None) -> Task:
        """Execute the agent (blocking) and return a completed or INPUT_REQUIRED Task."""
        task_id = str(uuid4())
        sid = _ensure_session_id(session_id)

        user_msg = Message.user(user_message)
        task = Task(
            id=task_id,
            session_id=sid,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[user_msg],
        )
        self._tasks[task_id] = task
        task.status = TaskStatus(state=TaskState.WORKING)

        adk_task = asyncio.create_task(self._run_adk(user_message, sid, task))
        self._running_tasks[task_id] = adk_task

        try:
            await adk_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error_msg = Message.agent(f"Error: {exc}")
            task.history.append(error_msg)
            task.status = TaskStatus(state=TaskState.FAILED, message=error_msg)
        finally:
            self._running_tasks.pop(task_id, None)

        return task

    async def resume(
        self,
        task_id: str,
        invocation_id: str,
        function_call_id: str,
        tool_name: str,
        response_payload: dict[str, Any],
    ) -> Task:
        """Resume a paused (INPUT_REQUIRED) task with the human's decision.

        Builds a FunctionResponse from the human's payload and passes it back
        to ADK using the same invocation_id, which continues the agent from
        exactly where it paused.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id!r} not found")
        if task.status.state != TaskState.INPUT_REQUIRED:
            raise ValueError(
                f"Task {task_id} is in state {task.status.state!r}, "
                f"expected {TaskState.INPUT_REQUIRED!r}"
            )

        sid = task.session_id or _ensure_session_id(None)

        # Record the human's decision in the task history
        decision = "approved" if response_payload.get("approved") else "rejected"
        comment = response_payload.get("comment", "")
        human_text = f"[Human {decision}] {comment}".strip()
        task.history.append(Message.user(human_text))
        task.status = TaskStatus(state=TaskState.WORKING)

        # Build the FunctionResponse that resumes the paused tool
        resume_content = genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        id=function_call_id,
                        name=tool_name,
                        response=response_payload,
                    )
                )
            ],
        )

        adk_task = asyncio.create_task(
            self._run_adk_resume(resume_content, sid, invocation_id, task)
        )
        self._running_tasks[task_id] = adk_task

        try:
            await adk_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error_msg = Message.agent(f"Error during resume: {exc}")
            task.history.append(error_msg)
            task.status = TaskStatus(state=TaskState.FAILED, message=error_msg)
        finally:
            self._running_tasks.pop(task_id, None)

        return task

    async def stream(
        self, user_message: str, session_id: str | None = None
    ) -> AsyncGenerator[Task, None]:
        """Stream task updates. Yields snapshots at tool calls and on INPUT_REQUIRED."""
        task_id = str(uuid4())
        sid = _ensure_session_id(session_id)

        user_msg = Message.user(user_message)
        task = Task(
            id=task_id,
            session_id=sid,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[user_msg],
        )
        self._tasks[task_id] = task
        yield task  # submitted

        await self._ensure_session(sid)

        event_q: asyncio.Queue[dict | None] = asyncio.Queue()

        async def _run_and_emit() -> None:
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            )
            try:
                async for event in self._runner.run_async(
                    user_id="default",
                    session_id=sid,
                    new_message=content,
                ):
                    if task_id in self._canceled:
                        break

                    # Long-running tool event → pause for human input
                    inv_id, fn_ids = _extract_long_running_info(event)
                    if fn_ids:
                        payload = _extract_tool_response_payload(event)
                        tool_name = _extract_tool_name_from_event(event, fn_ids[0])
                        await event_q.put({
                            "type": "input_required",
                            "invocation_id": inv_id,
                            "function_call_ids": fn_ids,
                            "tool_name": tool_name,
                            "pending_payload": payload,
                        })
                        break  # agent has paused; stop consuming events

                    # Tool call event — surface progress to SSE clients
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            fn_call = getattr(part, "function_call", None)
                            if fn_call:
                                await event_q.put({"type": "tool", "name": fn_call.name})

                    # Final text response
                    if event.is_final_response():
                        text = ""
                        if event.content and event.content.parts:
                            text = "".join(
                                p.text
                                for p in event.content.parts
                                if getattr(p, "text", None)
                            )
                        await event_q.put({"type": "final", "text": text})

            except asyncio.CancelledError:
                pass
            except Exception as exc:
                await event_q.put({"type": "error", "message": str(exc)})
            finally:
                await event_q.put(None)

        bg = asyncio.create_task(_run_and_emit())
        self._running_tasks[task_id] = bg

        task.status = TaskStatus(state=TaskState.WORKING)
        yield task  # working

        accumulated_text = ""
        try:
            while True:
                ev = await event_q.get()
                if ev is None:
                    break

                if task_id in self._canceled:
                    bg.cancel()
                    break

                t = ev["type"]

                if t == "input_required":
                    # Agent paused — surface approval request to human
                    task.metadata["pending_approval"] = {
                        "invocation_id": ev["invocation_id"],
                        "function_call_ids": ev["function_call_ids"],
                        "tool_name": ev["tool_name"],
                        "details": ev["pending_payload"],
                    }
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message=Message.agent(
                            ev["pending_payload"].get(
                                "message",
                                "Human approval required — see task metadata for details.",
                            )
                        ),
                    )
                    yield task
                    return  # SSE ends here; client polls or subscribes to resume

                elif t == "tool":
                    task.status = TaskStatus(
                        state=TaskState.WORKING,
                        message=Message.agent(f"Using tool: {ev['name']}…"),
                    )
                    yield task

                elif t == "final":
                    accumulated_text = ev["text"]

                elif t == "error":
                    raise RuntimeError(ev["message"])

        except RuntimeError as exc:
            error_msg = Message.agent(f"Error: {exc}")
            task.history.append(error_msg)
            task.status = TaskStatus(state=TaskState.FAILED, message=error_msg)
            yield task
            return

        finally:
            self._running_tasks.pop(task_id, None)

        if task_id in self._canceled:
            yield task
            return

        agent_msg = Message.agent(accumulated_text)
        task.history.append(agent_msg)
        task.artifacts = [
            Artifact(index=0, name="response", parts=[TextPart(text=accumulated_text)])
        ]
        task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg)
        yield task  # completed

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_adk(
        self, user_message: str, session_id: str, task: Task
    ) -> None:
        """Execute one turn and update *task* in place (completed or input_required)."""
        await self._ensure_session(session_id)
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        )
        await self._run_adk_content(content, session_id, None, task)

    async def _run_adk_resume(
        self,
        resume_content: genai_types.Content,
        session_id: str,
        invocation_id: str,
        task: Task,
    ) -> None:
        """Continue a paused invocation with the human's FunctionResponse."""
        await self._ensure_session(session_id)
        await self._run_adk_content(resume_content, session_id, invocation_id, task)

    async def _run_adk_content(
        self,
        content: genai_types.Content,
        session_id: str,
        invocation_id: str | None,
        task: Task,
    ) -> None:
        """Core execution loop — shared by fresh and resumed invocations."""
        kwargs: dict[str, Any] = dict(
            user_id="default",
            session_id=session_id,
            new_message=content,
        )
        if invocation_id:
            kwargs["invocation_id"] = invocation_id

        final_parts: list[str] = []
        last_event = None

        async for event in self._runner.run_async(**kwargs):
            last_event = event

            # Detect long-running tool pause
            inv_id, fn_ids = _extract_long_running_info(event)
            if fn_ids:
                payload = _extract_tool_response_payload(event)
                tool_name = _extract_tool_name_from_event(event, fn_ids[0])
                task.metadata["pending_approval"] = {
                    "invocation_id": inv_id,
                    "function_call_ids": fn_ids,
                    "tool_name": tool_name,
                    "details": payload,
                }
                task.status = TaskStatus(
                    state=TaskState.INPUT_REQUIRED,
                    message=Message.agent(
                        payload.get(
                            "message",
                            "Human approval required — see task metadata for details.",
                        )
                    ),
                )
                return  # agent paused

            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        final_parts.append(part.text)

        response_text = "\n".join(final_parts) if final_parts else "(no response)"
        agent_msg = Message.agent(response_text)
        task.history.append(agent_msg)
        task.artifacts = [
            Artifact(index=0, name="response", parts=[TextPart(text=response_text)])
        ]
        task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg)

    async def _ensure_session(self, session_id: str) -> None:
        existing = await self._session_service.get_session(
            app_name=_APP_NAME, user_id="default", session_id=session_id
        )
        if existing is None:
            await self._session_service.create_session(
                app_name=_APP_NAME, user_id="default", session_id=session_id
            )
