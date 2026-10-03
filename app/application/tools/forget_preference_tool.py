# -*- coding: utf-8 -*-
"""forget_preference_tool

长期记忆撤回路径：买家明确表示某条历史偏好不再适用（"以后不用避开塑料了"）时，
MainAgent 调它把该偏好从 Store 删掉，下一轮起不再注入。

与 remember_preference_tool 对称的两条纪律：
    1. buyer_id 从 ShoppingContext 取真实值，不信任模型入参；
    2. 只做精确 statement 匹配。删偏好不可逆，模糊匹配会误删
       （"不要塑料" 和 "不要塑料包装" 很像），未命中就把现存偏好回给模型让它用原文重试。

函数签名直接用于 LangChain 工具 schema。
"""
from app.application.runtime.results import ToolResult, ToolResultState
from app.infrastructure.semantic_memory import MemoryUnavailable

from app.domain.buyer.preference import PreferenceStore
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_forget_preference_tool(store: PreferenceStore, bus: TradeEventBus):
    async def forget_preference_tool(statement: str, memory_id: str = "", expected_version: int = 0) -> ToolResult:
        """删除买家的一条长期偏好（撤回后不再影响后续推荐）。

        仅在买家明确表示某条历史偏好不再适用时调用，例如"以后不用避开塑料了"。
        本轮的一次性例外（如"这次可以接受塑料"）不要调用，那属于临时要求。

        Args:
            memory_id (`str`): 最新记忆提示中的稳定 ID。
            expected_version (`int`): 最新记忆版本，不允许猜测。
            statement (`str`):
                要删除的偏好原文，必须与 <buyer-preferences> 里那一行的文字**完全一致**，
                如"不要塑料材质"。写错不会误删，工具会把现存偏好列出来供你重试。
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
            {"tool": "forget_preference_tool", "args": {"statement": statement}},
        )

        try:
            if getattr(store,"semantic_memory",False):
                if not memory_id or expected_version<1:raise ValueError("请从最新记忆列表读取 ID 和版本再操作")
                deleted=await store.delete_by_id(buyer_id,memory_id,expected_version,source_kind="agent",source_ref=session_id)
            else:deleted = await store.delete(buyer_id, statement)
        except ValueError as err:
            return ToolResult(data=f"[error] 撤回偏好失败：{err}", state=ToolResultState.ERROR,
                              error_code="unavailable" if isinstance(err, MemoryUnavailable) else "business_rejected",
                              error_reason="记忆服务不可用，撤回结果未确认" if isinstance(err,MemoryUnavailable) else "偏好撤回条件未满足，请核对当前记忆和版本")

        if deleted:
            return ToolResult(data=f"已撤回买家偏好：{statement}", state=ToolResultState.SUCCESS)

        # 未命中不算错误：把现存偏好回给模型，让它用原文重试，而不是让它以为删成功了
        remaining = await store.list_by_buyer(buyer_id)
        listing = (
            "\n".join(f"- [{p.kind}] {p.statement}" for p in remaining)
            if remaining
            else "（该买家当前没有任何长期偏好）"
        )
        return ToolResult(data=f"未找到偏好「{statement}」，没有删除任何内容。"
                        f"现存偏好如下，如需撤回请用其中一行的原文重试：\n{listing}", state=ToolResultState.SUCCESS)

    if getattr(store,"semantic_memory",False):
        implementation=forget_preference_tool
        async def forget_preference_tool(statement: str, memory_id: str, expected_version: int) -> ToolResult:
            return await implementation(statement,memory_id,expected_version)
        forget_preference_tool.__doc__=implementation.__doc__
    return forget_preference_tool
