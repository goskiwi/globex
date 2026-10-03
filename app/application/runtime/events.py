# -*- coding: utf-8 -*-
"""LangGraph 执行事件到 AG-UI 投影之间的稳定内部协议。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplyStart:
    type: str = "REPLY_START"


@dataclass(frozen=True)
class TextStart:
    block_id: str
    type: str = "TEXT_BLOCK_START"


@dataclass(frozen=True)
class TextDelta:
    block_id: str
    delta: str
    type: str = "TEXT_BLOCK_DELTA"


@dataclass(frozen=True)
class TextEnd:
    block_id: str
    type: str = "TEXT_BLOCK_END"


@dataclass(frozen=True)
class ToolEvent:
    type: str
    tool_call_id: str
    tool_call_name: str = ""
    delta: str = ""
    state: str = "success"
    data: Any = None
    arguments: dict | None = None
    failure: dict | None = None


@dataclass(frozen=True)
class ApprovalCall:
    id: str
    name: str
    input: str


@dataclass(frozen=True)
class RequireUserConfirm:
    reply_id: str
    tool_calls: list[ApprovalCall] = field(default_factory=list)
    type: str = "REQUIRE_USER_CONFIRM"


def approval_event(interrupts: tuple[Any, ...] | list[Any]) -> RequireUserConfirm | None:
    calls: list[ApprovalCall] = []
    reply_id = "langgraph"
    for item in interrupts:
        identifier = str(getattr(item, "id", ""))
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("action_id"):
            identifier += "." + str(value["action_id"])
        requests = value.get("action_requests", []) if isinstance(value, dict) else []
        for index, request in enumerate(requests):
            name = str(request.get("name", "tool"))
            calls.append(ApprovalCall(
                id=identifier or f"approval-{index}",
                name=name,
                input=__import__("json").dumps(request.get("args", {}), ensure_ascii=False),
            ))
    return RequireUserConfirm(reply_id, calls) if calls else None
