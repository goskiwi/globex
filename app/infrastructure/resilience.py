# -*- coding: utf-8 -*-
"""框架无关的工具超时配置、本地熔断注册表和 Redis 注册表调用接口。

执行包装只在 application/runtime/middleware.py 的原生工具中间件中实现。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# 工具超时分级（秒）：检索/知识库偏长，订单要快，子代理调度最宽松
DEFAULT_TIMEOUTS: dict[str, float] = {
    "product_search_tool": 15.0,
    "category_insight_tool": 15.0,
    "web_search_tool": 20.0,
    "create_order_tool": 10.0,
    "query_order_tool": 10.0,
    "cancel_order_tool": 10.0,
    "remember_preference_tool": 90.0,
    "update_preference_tool": 90.0,
    "task_dispatch": 180.0,
}


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: Optional[float] = None
    half_open_probing: bool = False

    @property
    def status(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.half_open_probing else "open"


@dataclass
class CircuitBreakerRegistry:
    """进程内共享的熔断状态表（按工具名）。"""

    failure_threshold: int = 3
    reset_seconds: float = 60.0
    _states: dict[str, _CircuitState] = field(default_factory=dict)

    def _state(self, tool_name: str) -> _CircuitState:
        return self._states.setdefault(tool_name, _CircuitState())

    def status(self, tool_name: str) -> str:
        return self._state(tool_name).status

    def allow(self, tool_name: str, now: Optional[float] = None) -> bool:
        """是否放行本次调用；冷却期满自动转半开并放行一次探测。"""
        state = self._state(tool_name)
        if state.opened_at is None:
            return True
        elapsed = (now or time.monotonic()) - state.opened_at
        if elapsed < self.reset_seconds:
            return False
        state.half_open_probing = True
        return True

    def record_success(self, tool_name: str) -> None:
        self._states[tool_name] = _CircuitState()

    def record_failure(self, tool_name: str, now: Optional[float] = None) -> None:
        state = self._state(tool_name)
        if state.half_open_probing:
            # 半开探测再次失败：重新打开并重置冷却计时
            state.opened_at = now or time.monotonic()
            state.half_open_probing = False
            return
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failure_threshold:
            state.opened_at = now or time.monotonic()


# ---- 注册表适配：共享实现的读写是异步的，本地实现是同步的 ----


async def _allow(registry: Any, tool_name: str) -> bool:
    if hasattr(registry, "allow_async"):
        return await registry.allow_async(tool_name)
    return registry.allow(tool_name)


async def _record_failure(registry: Any, tool_name: str) -> None:
    if hasattr(registry, "record_failure_async"):
        await registry.record_failure_async(tool_name)
        return
    registry.record_failure(tool_name)


async def _record_success(registry: Any, tool_name: str) -> None:
    if hasattr(registry, "record_success_async"):
        await registry.record_success_async(tool_name)
        return
    registry.record_success(tool_name)


async def _status(registry: Any, tool_name: str) -> str:
    if hasattr(registry, "status_async"):
        return await registry.status_async(tool_name)
    return registry.status(tool_name)
