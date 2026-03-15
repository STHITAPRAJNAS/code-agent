"""A2A protocol data models (JSON-RPC 2.0 + A2A spec)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Parts ─────────────────────────────────────────────────────────────────────

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class FilePart(BaseModel):
    type: Literal["file"] = "file"
    mime_type: str = "application/octet-stream"
    name: str = ""
    data: str = ""  # base64-encoded


class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)


Part = Union[TextPart, FilePart, DataPart]


# ── Message ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[Part]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", parts=[TextPart(text=text)])

    @classmethod
    def agent(cls, text: str) -> "Message":
        return cls(role="agent", parts=[TextPart(text=text)])

    def text(self) -> str:
        """Extract concatenated text from all TextPart instances."""
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))


# ── Task ─────────────────────────────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    INPUT_REQUIRED = "input-required"


class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[Message] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    index: int = 0
    name: Optional[str] = None
    description: Optional[str] = None
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    append: bool = False


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    status: TaskStatus
    history: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Agent Card ────────────────────────────────────────────────────────────────

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCapabilities(BaseModel):
    streaming: bool = True
    push_notifications: bool = False
    state_transition_history: bool = True


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default=["text"])
    default_output_modes: list[str] = Field(default=["text"])


# ── JSON-RPC 2.0 ──────────────────────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[dict[str, Any]] = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None


# ── JSON-RPC Error Codes (A2A spec) ───────────────────────────────────────────

class RpcError:
    PARSE_ERROR = JsonRpcError(code=-32700, message="Parse error")
    INVALID_REQUEST = JsonRpcError(code=-32600, message="Invalid Request")
    METHOD_NOT_FOUND = JsonRpcError(code=-32601, message="Method not found")
    INVALID_PARAMS = JsonRpcError(code=-32602, message="Invalid params")
    INTERNAL_ERROR = JsonRpcError(code=-32603, message="Internal error")
    TASK_NOT_FOUND = JsonRpcError(code=-32001, message="Task not found")
    TASK_CANCELED = JsonRpcError(code=-32002, message="Task canceled")
    TASK_NOT_CANCELABLE = JsonRpcError(code=-32003, message="Task not cancelable")
