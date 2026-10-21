# -*- coding: utf-8 -*-
"""运行内部事件总线；内部消费者读取原始事实，公开序列化采用统一副本。"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.infrastructure.security.public_projection import PublicProjection
from app.infrastructure.tracing import current_correlation
from app.infrastructure.operational_metrics import observe_event

TradeEventType = str

# 只观察当前执行任务及其子协程，避免同会话排队时混入另一轮的商品结果。
_run_observer: ContextVar[Callable[["TradeEvent"], None] | None] = ContextVar(
    "globex_trade_event_observer", default=None,
)


@contextmanager
def observe_run_events(observer: Callable[["TradeEvent"], None]):
    previous = _run_observer.get()
    def notify(event):
        if previous is not None:
            previous(event)
        observer(event)
    token = _run_observer.set(notify)
    try:
        yield
    finally:
        _run_observer.reset(token)

EVENT_TYPES = (
    "presentation.result",
    "agent.dispatch",
    "tool.invoke",
    "tool.result",
    "handoff.result",
    "plan.update",
    "context.compressed",
    "model.fallback",
    "usage.summary",
    "cache.hit",
    "task.queued",
    "task.started",
    "confirmation.required",
    "confirmation.resolved",
    "final.result",
    "error",
)


@dataclass(frozen=True)
class TradeEvent:
    shopping_session_id: str
    type: TradeEventType
    payload: Any
    occurred_at: str
    correlation: dict[str, str] = field(default_factory=dict)
    public_payload: Any = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "shopping_session_id": self.shopping_session_id,
            "type": self.type,
            "payload": self.public_payload if self.public_payload is not None else self.payload,
            "occurred_at": self.occurred_at,
            "correlation": self.correlation,
        }

    @staticmethod
    def from_dict(raw: dict) -> "TradeEvent":
        return TradeEvent(
            shopping_session_id=raw["shopping_session_id"],
            type=raw["type"],
            payload=raw.get("payload"),
            occurred_at=raw.get("occurred_at", ""),
            correlation=raw.get("correlation", {}),
        )


@dataclass
class TradeEventBus:
    """进程内事实分发；跨进程恢复由持久运行日志提供。"""

    projection: PublicProjection = field(default_factory=PublicProjection)
    _subscribers: dict[str, list[asyncio.Queue]] = field(default_factory=dict)

    def subscribe(self, shopping_session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(shopping_session_id, []).append(queue)
        return queue

    def unsubscribe(self, shopping_session_id: str, queue: asyncio.Queue) -> None:
        queues = self._subscribers.get(shopping_session_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(shopping_session_id, None)

    def publish(self, shopping_session_id: str, event_type: TradeEventType, payload: Any) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型：{event_type}")
        event = TradeEvent(
            shopping_session_id=shopping_session_id,
            type=event_type,
            payload=payload,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            correlation=current_correlation(shopping_session_id),
            public_payload=self.projection.trade(event_type,payload),
        )
        observe_event(event_type, payload)
        observer = _run_observer.get()
        if observer is not None:
            observer(event)
        for queue in self._subscribers.get(event.shopping_session_id, []):
            queue.put_nowait(event)
