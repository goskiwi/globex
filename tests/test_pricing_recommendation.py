"""完整报价与推荐链：使用真实目录、证据库和 SQLite，不调用外部模型。"""
import pytest

from app.application.tools.recommendation_tools import build_recommendation_tool, Pick
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.pricing import QuoteItem
from app.application.usecases.order_usecases import OrderItemInput
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env, test_address as address


@pytest.mark.asyncio
async def test_quote_confirmation_order_and_cancel_share_amounts(confirmation_env):
    env = confirmation_env
    pricing = env.service._pricing
    quote = await pricing.quote([QuoteItem("P1001", "P1001-S1", 1), QuoteItem("P1001", "P1001-S1", 1)], "CN", "USD")
    assert len(quote["items"]) == 1
    assert quote["items"][0]["quantity"] == 2
    prepared = (await env.service.prepare_order("buyer-1", "session-1", [OrderItemInput("P1001", "P1001-S1", 2)], address(), "USD"))["confirmation"]
    assert all(prepared["payload"][key] == value for key, value in quote.items())
    approved = (await env.service.resolve(prepared["confirmation_id"], "buyer-1", "session-1", prepared["snapshot_hash"], True))["order"]
    assert approved["pricing"] == quote
    assert approved["total_amount_minor"] == quote["total_amount_minor"]
    assert approved["total_amount_major"] == quote["total_amount_minor"] / 100
    cancellation = (await env.service.prepare_cancel("buyer-1", "session-1", approved["order_id"], "测试"))["confirmation"]
    assert cancellation["payload"]["total_amount_minor"] == quote["total_amount_minor"]
    replay = await env.service.resolve(prepared["confirmation_id"], "buyer-1", "session-1", prepared["snapshot_hash"], True)
    assert replay["order"] == approved


@pytest.mark.asyncio
async def test_search_and_multi_quantity_share_pricing(confirmation_env):
    env = confirmation_env
    catalog = CatalogSearchUseCase(env.products, pricing=env.service._pricing)
    result = await catalog.execute(ProductSearchSpec(sku_id="P1001-S1", ship_to="CN"))
    single = await catalog.pricing.quote([QuoteItem("P1001", "P1001-S1", 1)], "CN", "CNY")
    assert result["hits"][0]["landed_price"] == single
    double = await catalog.pricing.quote([QuoteItem("P1001", "P1001-S1", 2)], "CN", "CNY")
    assert double["subtotal_minor"] == single["subtotal_minor"] * 2
    assert double["freight_minor"] == 4000
    assert double["total_amount_minor"] == 41800
    with pytest.raises(ValueError):
        await catalog.pricing.quote([QuoteItem("P1001", "P1001-S1", 1)], "XX", "CNY")


@pytest.mark.asyncio
async def test_recommendation_uses_evidence_current_constraints_and_actual_quantities(confirmation_env):
    env = confirmation_env
    catalog = CatalogSearchUseCase(env.products, pricing=env.service._pricing)
    result = await catalog.execute(ProductSearchSpec(product_id="P1001", ship_to="CN"))
    await env.evidence.save("buyer-1", "session-1", "products", result)
    tool = build_recommendation_tool(catalog, env.evidence, env.bus)
    pick = Pick(product_id="P1001", sku_id="P1001-S1", quantity=2, reason="适合旅行")
    context = ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY",
        effective_search={"parameters": {"ship_to": "CN", "target_currency": "CNY", "landed_budget_major": 400}})
    token = ShoppingContext.set(context)
    try:
        rejected = await tool([pick], "bundle", None, [], "两件搭配用于旅行。")
        assert not rejected.ok and "超过当前预算" in str(rejected)
        assert not await env.evidence.search("buyer-1", "session-1", kind="recommendation")
    finally:
        ShoppingContext.reset(token)

    token = ShoppingContext.set(ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY",
        effective_search={"parameters": {"ship_to": "CN", "target_currency": "CNY", "landed_budget_major": 450}}))
    try:
        accepted = await tool([pick], "bundle", None, [], "两件搭配用于旅行。")
        assert accepted.ok
        saved = (await env.evidence.search("buyer-1", "session-1", kind="recommendation"))[0]["data"]
        assert saved["quote"]["total_amount_minor"] == 41800
        assert saved["hits"][0]["landed_price"]["items"][0]["quantity"] == 2
        assert ShoppingContext.current().selected_lines == ()
        assert await env.store.list_confirmations(buyer_id="buyer-1", session_id="session-1") == []
    finally:
        ShoppingContext.reset(token)


def test_final_recommendation_is_not_replaced_by_later_search():
    from ag_ui.core import RunAgentInput
    from app.application.agents.ag_ui_adapter import AGUIRunAdapter
    from app.infrastructure.eventbus import TradeEvent
    request = RunAgentInput(thread_id="s", run_id="r", state={}, messages=[], tools=[], context=[], forwarded_props={})
    adapter = AGUIRunAdapter(request, lambda event: None)
    recommendation = {"mode": "alternatives", "hits": [{"product_id": "P1001"}], "quote": None, "result_ref": "ctx_test"}
    adapter.on_trade_event(TradeEvent("s", "recommendation.result", recommendation, ""))
    adapter.on_trade_event(TradeEvent("s", "tool.result", {"tool": "product_search_tool", "hits": [{"product_id": "P1003"}]}, ""))
    assert adapter.state["recommendation"] is None
    adapter.finish("已交付推荐", "completed", None, product_delivery_complete=True)
    assert adapter.state["recommendation"] == recommendation
    assert "products" not in adapter.state


def test_schema_migration_preserves_nonempty_history(tmp_path):
    import sqlite3
    from scripts.migrate_pricing_schema import migrate
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO orders VALUES ('historical')")
    with pytest.raises(ValueError, match="未修改数据库"):
        migrate(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
        assert [r[1] for r in db.execute("PRAGMA table_info(orders)")] == ["order_id"]


@pytest.mark.asyncio
async def test_native_tool_accepts_typed_picks_and_preserves_result(confirmation_env):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.application.runtime.tools import as_langchain_tool
    from app.application.tools.recommendation_tools import RecommendationInput
    from app.application.runtime.delivery import FinalDeliveryMiddleware
    env = confirmation_env
    catalog = CatalogSearchUseCase(env.products, pricing=env.service._pricing)
    await env.evidence.save("buyer-1", "session-1", "products",
        await catalog.execute(ProductSearchSpec(product_id="P1001")))
    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    model = Model(responses=[AIMessage(content="", tool_calls=[{"id": "recommend-1", "name": "recommend_products", "args": {
        "mode": "bundle", "guidance":"两件用于分开收纳旅行用品，费用请查看具体报价。", "preferred_sku_id":None, "dimensions":["旅行收纳"], "picks": [{"product_id": "P1001", "sku_id": "P1001-S1", "quantity": 2, "reason": "适合旅行"}]}}]),
        AIMessage(content="两件合计418元")])
    tool = as_langchain_tool(build_recommendation_tool(catalog, env.evidence, env.bus), args_schema=RecommendationInput)
    graph = create_agent(model, tools=[tool], middleware=[FinalDeliveryMiddleware()])
    token = ShoppingContext.set(ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY",
        effective_search={"parameters": {"ship_to": "CN", "target_currency": "CNY"}}))
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(content="推荐两件") ]})
        outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(outputs) == 1 and outputs[0].status == "success"
        assert '41800' in outputs[0].content
        assert model.i == 1  # 推荐后没有第二次模型调用。
        assert result["messages"][-1].content == "两件用于分开收纳旅行用品，费用请查看具体报价。"
    finally:
        ShoppingContext.reset(token)


@pytest.mark.asyncio
async def test_cancel_does_not_restore_mutated_order_quantities(confirmation_env):
    from sqlalchemy import update
    from app.infrastructure.persistence.sql.tables import OrderLineRow
    env = confirmation_env
    c = (await env.service.prepare_order("buyer-1", "session-1", [OrderItemInput("P1001", "P1001-S1", 1)], address()))["confirmation"]
    order = (await env.service.resolve(c["confirmation_id"], "buyer-1", "session-1", c["snapshot_hash"], True))["order"]
    cancel = (await env.service.prepare_cancel("buyer-1", "session-1", order["order_id"], "测试"))["confirmation"]
    stock = await env.store.get_inventory()
    async with env.engine.begin() as db:
        await db.execute(update(OrderLineRow).where(OrderLineRow.order_id == order["order_id"]).values(quantity=10))
    with pytest.raises(ValueError, match="订单明细"):
        await env.service.resolve(cancel["confirmation_id"], "buyer-1", "session-1", cancel["snapshot_hash"], True)
    assert await env.store.get_inventory() == stock


@pytest.mark.asyncio
async def test_different_skus_and_source_currencies_use_one_total():
    from app.domain.catalog.product import Product
    from app.domain.catalog.sku import Sku
    from app.domain.catalog.money import Money
    from app.application.usecases.pricing import PricingService
    from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
    product = Product(product_id="P9001", title="测试商品", brand="test", category="数码配件", origin_country="CN",
        description="合成夹具", ships_to=["CN"], skus=[
            Sku("P9001-S1", "标准", Money.of(10000, "CNY"), 10),
            Sku("P9001-S2", "高配", Money.of(2000, "USD"), 10)])
    service = PricingService(InMemoryProductRepository([product]))
    quote = await service.quote([QuoteItem("P9001", "P9001-S1", 1), QuoteItem("P9001", "P9001-S2", 2)], "CN", "CNY")
    assert quote["subtotal_minor"] == 38400
    assert quote["freight_minor"] == 6500
    assert quote["total_amount_minor"] == 44900
    assert quote["items"][1]["source_currency"] == "USD"
    assert quote["items"][1]["unit_price_minor"] == 14200
