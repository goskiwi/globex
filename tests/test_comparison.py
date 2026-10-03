"""比较复用权威报价与原生图交付，不修改买家选择和交易。"""
import json
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.application.tools.recommendation_tools import build_comparison_tool, ComparisonInput, Pick
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.delivery import FinalDeliveryMiddleware
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env


@pytest.fixture
async def comparison_env(confirmation_env):
    env = confirmation_env
    catalog = CatalogSearchUseCase(env.products, pricing=env.service._pricing)
    await env.evidence.save("buyer-1", "session-1", "products", await catalog.execute(ProductSearchSpec(product_id="P1001")))
    env.tool = build_comparison_tool(catalog, env.evidence, env.bus)
    env.guidance = "两款的用途相同，按颜色偏好选择即可。"
    env.entries = [Pick(product_id="P1001", sku_id=sku, quantity=2, reason="颜色偏好", tradeoffs=["颜色不同"])
                   for sku in ("P1001-S1", "P1001-S2")]
    return env


@pytest.mark.asyncio
async def test_compare_sibling_skus_quotes_and_terminal_delivery(comparison_env):
    env = comparison_env
    token = ShoppingContext.set(ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY",
        effective_search={"parameters": {"ship_to": "CN", "target_currency": "CNY"}}))
    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs): return self
    model = Model(responses=[AIMessage(content="", tool_calls=[{"id":"comparison", "name":"compare_products", "args":{
        "entries":[p.model_dump() for p in env.entries], "guidance":env.guidance, "preferred_sku_id":"P1001-S1", "dimensions":["颜色偏好"]}}]), AIMessage(content="不该执行")])
    graph = create_agent(model, tools=[as_langchain_tool(env.tool, args_schema=ComparisonInput)], middleware=[FinalDeliveryMiddleware()])
    try:
        output = await graph.ainvoke({"messages":[HumanMessage(content="比较两种规格")]})
        assert model.i == 1 and output["messages"][-1].content == env.guidance
        receipt = next(m for m in output["messages"] if isinstance(m, ToolMessage))
        assert receipt.status == "success"
        result = json.loads(receipt.content)
        assert [c["default_sku_id"] for c in result["hits"]] == ["P1001-S1", "P1001-S2"]
        assert [c["landed_price"]["total_amount_minor"] for c in result["hits"]] == [41800,43800]
        assert result['dimensions'] == ['颜色偏好']
        assert result['max_items'] == 12
        assert ShoppingContext.current().selected_lines == ()
        assert await env.store.list_confirmations(buyer_id="buyer-1", session_id="session-1") == []
        assert await env.evidence.get("buyer-2", "session-1", result["result_ref"]) is None
    finally:
        ShoppingContext.reset(token)


@pytest.mark.asyncio
async def test_unknown_destination_does_not_invent_quote(comparison_env):
    env = comparison_env
    token = ShoppingContext.set(ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY"))
    try:
        result = json.loads((await env.tool(env.entries, None, [], env.guidance)).text)
        assert result["preferred_sku_id"] is None
        assert all("landed_price" not in c for c in result["hits"])
    finally:
        ShoppingContext.reset(token)


@pytest.mark.asyncio
async def test_over_budget_is_visible_but_not_preferred(comparison_env):
    env = comparison_env
    token = ShoppingContext.set(ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY",
        effective_search={"parameters": {"ship_to":"CN", "landed_budget_major":430, "target_currency":"CNY"}}))
    try:
        result = json.loads((await env.tool(env.entries, "P1001-S1", ["颜色偏好"], env.guidance)).text)
        assert result["hits"][1]["constraint_issues"] == ["over_landed_budget"]
        assert not (await env.tool(env.entries, "P1001-S2", ["颜色偏好"], env.guidance)).ok
        assert not (await env.tool(env.entries, "P9999-S1", [], env.guidance)).ok
        assert not (await env.tool([env.entries[0], env.entries[0]], None, [], env.guidance)).ok
    finally:
        ShoppingContext.reset(token)


@pytest.mark.asyncio
async def test_comparison_requires_current_buyer_evidence(comparison_env):
    token = ShoppingContext.set(ShoppingContextSnapshot("session-1", "other-buyer", "zh-CN", "CNY"))
    try:
        assert not (await comparison_env.tool(comparison_env.entries, None, [], comparison_env.guidance)).ok
    finally:
        ShoppingContext.reset(token)


def test_comparison_projection_survives_later_search_and_restores():
    from ag_ui.core import RunAgentInput
    from app.application.agents.ag_ui_adapter import AGUIRunAdapter
    from app.infrastructure.eventbus import TradeEvent
    request = RunAgentInput(thread_id="s", run_id="r", state={}, messages=[], tools=[], context=[], forwarded_props={})
    adapter = AGUIRunAdapter(request, lambda e: None)
    result = {"hits":[{"product_id":"P1001", "default_sku_id":"P1001-S1"}, {"product_id":"P1001", "default_sku_id":"P1001-S2"}], "preferred_sku_id":None,"dimensions":["颜色"],"max_items":12}
    adapter.on_trade_event(TradeEvent("s","recommendation.result",{"hits":[]},""))
    adapter.on_trade_event(TradeEvent("s","comparison.result",result,""))
    adapter.on_trade_event(TradeEvent("s","tool.result",{"tool":"product_search_tool","hits":[]},""))
    assert adapter.state["comparison"] is None
    adapter.finish("已交付比较", "completed", None, product_delivery_complete=True)
    assert adapter.state["comparison"] == result and adapter.state["recommendation"] is None
    restored = AGUIRunAdapter(request, lambda e: None, authoritative_state=adapter.state)
    assert restored.state["comparison"] == result


@pytest.mark.asyncio
async def test_comparison_survives_journal_restart_and_followup(tmp_path):
    from app.infrastructure.ag_ui_journal import AGUIJournal
    from tests.test_ag_ui_journal import body
    path = tmp_path / "comparison.db"
    journal = AGUIJournal(path)
    result = {"hits":[{"product_id":"P1001","default_sku_id":"P1001-S1"},
                      {"product_id":"P1001","default_sku_id":"P1001-S2"}], "preferred_sku_id":None,"dimensions":["颜色"],"max_items":12}
    await journal.reserve(body(), "b1", "owner")
    await journal.append("r1", "owner", [{"type":"STATE_SNAPSHOT","snapshot":{"comparison":result,"products":[],"recommendation":None}},
        {"type":"RUN_FINISHED","threadId":"s1","runId":"r1"}])
    reopened = AGUIJournal(path)
    assert (await reopened.session("s1", "b1"))["run"]["state"]["comparison"] == result
    await reopened.reserve(body("r2"), "b1", "owner2")
    followup = await reopened.session("s1", "b1")
    assert followup["run"]["state"]["comparison"] == result
    assert followup["run"]["state"]["deliveredRunId"] == "r1"
    assert followup["productHistory"] == []


@pytest.mark.asyncio
async def test_parallel_final_presentations_do_not_race_to_overwrite_state():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    last = AIMessage(content="", tool_calls=[{"id":"a","name":"recommend_products","args":{}},
                                            {"id":"b","name":"compare_products","args":{}}])
    handler = AsyncMock()
    middleware = FinalDeliveryMiddleware()
    for call in last.tool_calls:
        result = await middleware.awrap_tool_call(SimpleNamespace(tool_call=call, state={"messages":[last]}), handler)
        assert result.status == "error"
    handler.assert_not_called()
