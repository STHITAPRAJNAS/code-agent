"""Async task queue — fire-and-forget execution for long-running agent tasks.

ADK itself has no built-in task queue.  The recommended pattern for A2A
long-running work is:

  1. Client POSTs tasks/submit → gets back a Task in SUBMITTED state immediately
  2. TaskQueue workers pick the job up from an asyncio.Queue in the background
  3. Client polls tasks/get or opens tasks/sendSubscribe SSE to watch progress

This allows the HTTP server to remain responsive while the LLM works.
Workers are started once at app startup (via the FastAPI lifespan) and
stopped cleanly on shutdown.

Usage
-----
    # in FastAPI lifespan:
    queue = TaskQueue(runner, num_workers=4)
    await queue.start()
    ...
    await queue.stop()

    # in a route handler:
    task = await queue.submit("Implement PROJ-123", session_id="abc")
    return {"task_id": task.id}
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from code_agent.a2a.models import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

logger = logging.getLogger(__name__)

_SENTINEL = object()  # signals workers to shut down


class TaskQueue:
    """Asyncio-based worker pool that executes agent tasks in the background.

    Parameters
    ----------
    runner:
        The AgentRunner instance that executes ADK invocations.
    num_workers:
        Number of concurrent worker coroutines (default 4).
    """

    def __init__(self, runner: "AgentRunner", num_workers: int = 4) -> None:  # noqa: F821
        self._runner = runner
        self._num_workers = num_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn worker coroutines.  Call once at application startup."""
        for i in range(self._num_workers):
            t = asyncio.create_task(self._worker(i), name=f"task-queue-worker-{i}")
            self._worker_tasks.append(t)
        logger.info("TaskQueue started with %d worker(s)", self._num_workers)

    async def stop(self) -> None:
        """Send shutdown sentinels and wait for workers to drain."""
        for _ in self._worker_tasks:
            await self._queue.put(_SENTINEL)
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("TaskQueue stopped")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def submit(
        self,
        user_message: str,
        session_id: str | None = None,
    ) -> Task:
        """Enqueue a task and return immediately with SUBMITTED state.

        The caller can use the returned task.id to poll via tasks/get or
        subscribe via tasks/sendSubscribe.
        """
        sid = session_id or str(uuid4())
        user_msg = Message.user(user_message)
        task = Task(
            id=str(uuid4()),
            session_id=sid,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[user_msg],
        )
        # Register in the runner's shared task store so tasks/get works
        self._runner._tasks[task.id] = task
        await self._queue.put((task.id, user_message, sid))
        logger.info("Task %s submitted (queue depth=%d)", task.id, self._queue.qsize())
        return task

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """Pull tasks from the queue and execute them one at a time."""
        logger.debug("Worker %d ready", worker_id)
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break

            task_id, user_message, session_id = item
            task = self._runner._tasks.get(task_id)

            if task is None:
                logger.warning("Worker %d: task %s not found — skipping", worker_id, task_id)
                self._queue.task_done()
                continue

            if task_id in self._runner._canceled:
                logger.info("Worker %d: task %s was canceled before execution", worker_id, task_id)
                self._queue.task_done()
                continue

            logger.info("Worker %d: executing task %s", worker_id, task_id)
            task.status = TaskStatus(state=TaskState.WORKING)

            try:
                # Create a cancellable asyncio.Task so cancel_task() can
                # interrupt mid-flight LLM execution.
                adk_task = asyncio.create_task(
                    self._runner._run_adk(user_message, session_id)
                )
                self._runner._running_tasks[task_id] = adk_task

                try:
                    response_text = await adk_task
                except asyncio.CancelledError:
                    logger.info("Worker %d: task %s cancelled during execution", worker_id, task_id)
                    # status already set to CANCELED by cancel_task()
                    continue

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
                logger.info("Worker %d: task %s completed", worker_id, task_id)

            except Exception as exc:
                logger.exception("Worker %d: task %s failed: %s", worker_id, task_id, exc)
                error_msg = Message.agent(f"Error: {exc}")
                task.history.append(error_msg)
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message=error_msg,
                )
            finally:
                self._runner._running_tasks.pop(task_id, None)
                self._queue.task_done()
