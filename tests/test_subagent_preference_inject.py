"""子 Agent 只接收一份生效条件；偏好、例外和用户归属来自服务端。"""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from langchain_core.messages import HumanMessage
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.application.agents.shopping_state import ShoppingWork, Filters
from app.domain.buyer.preference import BuyerPreference, MaterialExclusion
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus

class RecordingFactory:
    def __init__(self):
        self.seen=[]
        self.contexts=[]
    def build(self):
        owner=self
        class Graph:
            async def ainvoke(self, inputs, **kwargs):
                owner.seen.append(json.loads(inputs["messages"][-1].content))
                owner.contexts.append(ShoppingContext.current())
                return {"handoff_result":{"status":"needs_input","summary":"请补充场景","questions":["用途？"]}}
        return Graph()

@pytest.mark.parametrize("role", ["search_agent","trade_agent"])
async def test_preferences_are_from_snapshot_not_model_transcription(role):
    facts=(BuyerPreference(buyer_id="b",kind="dislike",statement="不要合成聚合物",
                           constraint=MaterialExclusion(("合成聚合物",)), evidence="不要合成聚合物"),)
    snapshot=ShoppingContextSnapshot("s","b","zh-CN","CNY",preference_facts=facts)
    token=ShoppingContext.set(snapshot)
    factory=RecordingFactory()
    try:
        await build_task_dispatch_tool(factory,factory,TradeEventBus())(role,{"goal":"研究"})
        params=factory.seen[0]["parent_context"]["effective_search"]["parameters"]
        assert params["excluded_material_tags"]==["合成聚合物"]
        assert ShoppingContext.current() is snapshot
        assert "shopping_work" not in factory.seen[0]["parent_context"]
    finally:
        ShoppingContext.reset(token)

async def test_explicit_exception_is_kept_in_delegation():
    pref=BuyerPreference(buyer_id="b",kind="dislike",statement="不要合成聚合物",
                         constraint=MaterialExclusion(("合成聚合物",)), evidence="不要合成聚合物")
    token=ShoppingContext.set(ShoppingContextSnapshot("s","b","zh-CN","CNY",preference_facts=(pref,)))
    factory=RecordingFactory()
    work=ShoppingWork(filters=Filters(ship_to="CN",target_currency="CNY"),ignored_preferences=[pref.statement])
    try:
        await build_task_dispatch_tool(factory,factory,TradeEventBus())("search_agent",{"goal":"研究","filters":{"price_max_major":300}},
            runtime=SimpleNamespace(state={"shopping_work":work.model_dump()},tool_call_id="task"))
        params=factory.seen[0]["parent_context"]["effective_search"]["parameters"]
        assert params["ship_to"]=="CN" and params["price_max_major"]==300
        assert params["excluded_material_tags"]==[]
    finally:
        ShoppingContext.reset(token)

async def test_scope_cannot_drop_inherited_material_or_unverified_requirements():
    token=ShoppingContext.set(ShoppingContextSnapshot("s","b","zh-CN","CNY"))
    factory=RecordingFactory()
    work=ShoppingWork(filters=Filters(excluded_material_tags=["金属"]),unverified_requirements=["不含尼龙"])
    try:
        await build_task_dispatch_tool(factory,factory,TradeEventBus())("search_agent",
            {"goal":"研究","filters":{"excluded_material_tags":[]},"requirements":["防水"]},
            runtime=SimpleNamespace(state={"shopping_work":work.model_dump(),
                "messages":[HumanMessage(name="b",content="我要防水的背包")]},tool_call_id="task"))
        effective=factory.seen[0]["parent_context"]["effective_search"]
        assert effective["parameters"]["excluded_material_tags"]==["金属"]
        assert effective["unverified_requirements"]==["不含尼龙","防水"]
        assert "filters" not in factory.seen[0]["delegated_task"]
    finally:
        ShoppingContext.reset(token)

async def test_missing_context_does_not_dispatch():
    token=ShoppingContext.set(None)
    factory=RecordingFactory()
    try:
        result=(await build_task_dispatch_tool(factory,factory,TradeEventBus())("search_agent",{"goal":"研究"})).data
        assert result["status"]=="failed" and not factory.seen
    finally:
        ShoppingContext.reset(token)

async def test_wrong_owner_does_not_dispatch():
    token=ShoppingContext.set(ShoppingContextSnapshot("s","b","zh-CN","CNY"))
    factory=RecordingFactory()
    try:
        result=(await build_task_dispatch_tool(factory,factory,TradeEventBus())("search_agent",{"goal":"研究"},
            runtime=SimpleNamespace(state={"shopping_work_owner":"other"},tool_call_id="task"))).data
        assert result["status"]=="failed" and not factory.seen
    finally:
        ShoppingContext.reset(token)
