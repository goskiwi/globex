# -*- coding: utf-8 -*-
"""remember_preference_tool

长期记忆写路径：MainAgent 在对话中发现买家的稳定偏好（材质忌口、风格取向、
预算习惯等）时调用，跨会话持久化；读路径由运行时在每次模型决策前统一解析并投影。

买家身份从 ShoppingContext 取真实值，不信任模型入参。

函数签名直接用于 LangChain 工具 schema。
"""
from typing import Literal

from app.application.runtime.results import ToolResult, ToolResultState
from app.infrastructure.semantic_memory import MemoryUnavailable

from app.domain.buyer.preference import BuyerPreference, PreferenceStore
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_remember_preference_tool(store: PreferenceStore, bus: TradeEventBus):
    async def remember_preference_tool(
        kind: Literal["like", "dislike"],
        statement: str,
    ) -> ToolResult:
        """记住买家的一条长期偏好（跨会话生效）。仅在买家表达出稳定偏好时调用，
        一次性的临时要求（如"这次要军绿色"）不要记。

        Args:
            kind (`str`):
                "like"（正向偏好，如"喜欢小众设计"）或 "dislike"（忌口/黑名单，如"不要塑料材质"）。
            statement (`str`):
                一句话偏好陈述，10 字以内最佳，如"不要塑料材质"。
        """
        snapshot = ShoppingContext.current()
        if snapshot is None:
            return ToolResult(data="[error] 缺少可信买家上下文", state=ToolResultState.ERROR,
                              error_code="business_rejected", error_reason="缺少可信买家上下文")
        buyer_id = snapshot.buyer_id
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "remember_preference_tool", "args": {"kind": kind, "statement": statement}},
        )
        try:
            normalized = await store.append(BuyerPreference(buyer_id=buyer_id, kind=kind, statement=statement, source_kind="agent", source_ref=session_id))
        except ValueError as err:
            return ToolResult(data=f"[error] {err}", state=ToolResultState.ERROR,
                              error_code="unavailable" if isinstance(err, MemoryUnavailable) else "business_rejected",
                              error_reason="记忆服务不可用，保存结果未确认" if isinstance(err,MemoryUnavailable) else "偏好未满足保存条件")
        statement = "；".join(p.statement for p in normalized) if normalized is not None else statement
        return ToolResult(data=f"已记住买家偏好：[{kind}] {statement}", state=ToolResultState.SUCCESS)

    return remember_preference_tool
