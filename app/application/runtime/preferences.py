"""在图决策边界刷新偏好；同一模型决策及其工具批次使用同一份事实。"""
from dataclasses import asdict, replace
from typing_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from app.application.memory.preference_resolution import resolve_preferences
from app.domain.buyer.preference import BuyerPreference
from app.infrastructure.context import ShoppingContext


class PreferenceState(AgentState):
    preference_snapshot: NotRequired[dict]


class PreferenceStateMiddleware(AgentMiddleware):
    state_schema = PreferenceState

    def __init__(self, store, selector, top_k):
        self.store, self.selector, self.top_k = store, selector, top_k

    async def abefore_model(self, state, runtime):
        context = ShoppingContext.current()
        if context is None:
            raise ValueError("偏好解析缺少可信买家上下文")
        query = next((m.text for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage) and m.name == context.buyer_id), "")
        resolved = await resolve_preferences(self.store, self.selector, context.buyer_id,
                                             query, self.top_k)
        removed = [RemoveMessage(id=m.id) for m in state["messages"]
                   if m.name == "memory_hint" and m.id != "current-preferences"]
        return {"preference_snapshot": {"buyer_id": context.buyer_id,
                                         "facts": [asdict(p) for p in resolved.facts]},
                "messages": [*removed, SystemMessage(id="current-preferences",
                    name="memory_hint", content=resolved.hint)]}

    async def awrap_tool_call(self, request, handler):
        context = ShoppingContext.current()
        snapshot = request.state["preference_snapshot"]
        if context is None or snapshot["buyer_id"] != context.buyer_id:
            raise ValueError("偏好快照与当前买家不一致")
        token = ShoppingContext.set(replace(context,
            preference_facts=tuple(BuyerPreference(**p) for p in snapshot["facts"])))
        try:
            return await handler(request)
        finally:
            ShoppingContext.reset(token)
