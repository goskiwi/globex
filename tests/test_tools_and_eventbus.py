# -*- coding: utf-8 -*-
"""工具层与事件总线单测：工具直调（绕过 LLM）+ EventBus 订阅。"""
from app.application.agents.shopping_state import Filters
from tests.shopping_state_helpers import run_search
import asyncio
import json

import pytest

from app.application.tools.order_tools import build_create_order_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.order_usecases import PlaceOrderUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryProductRepository,
)
from tests.trade_test_helpers import confirmation_env  # noqa: F401

ADDRESS = {
    "recipient_name": "张三",
    "country": "CN",
    "state": "浙江",
    "city": "杭州",
    "address_line": "西湖区某路 1 号",
    "postal_code": "310000",
    "phone": "13800000000",
}


class TestTradeEventBus:
    async def test_publish_routes_to_subscriber(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        other = bus.subscribe("s2")
        bus.publish("s1", "final.result", {"text": "done"})

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "final.result"
        assert other.empty(), "事件不能串台到其他会话"

    def test_reject_unknown_event_type(self):
        bus = TradeEventBus()
        with pytest.raises(ValueError, match="未知事件类型"):
            bus.publish("s1", "not.a.type", {})


class TestToolsDirectInvoke:
    async def test_product_search_tool(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await run_search(tool, normalized_query="旅行三件套 抗造")
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.text)
        assert payload["hits"][0]["product_id"] == "P1001"
        # 业务函数发布调用条件；结果事件由运行时发布
        assert queue.qsize() == 1

    async def test_product_search_tool_accepts_numeric_parameters(self):
        """明确的数值参数进入检索链，业务代码不再做模型专用字符串转换。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await run_search(tool, normalized_query="旅行三件套 抗造", top_k=3, filters=Filters(price_max_major=300, target_currency="CNY"))
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.text)
        # 预算硬约束生效：候选主价均不超过 300。
        assert payload["hits"], "明确的数值价格上限可以执行检索"
        for hit in payload["hits"]:
            assert hit["price_major"] <= 300
        # 业务函数发布调用条件；结果事件由运行时发布
        assert queue.qsize() == 1

    async def test_product_search_tool_excludes_synthetic_polymer(self):
        """“不要塑料”必须作为结构化材质约束进入工具调用，而非事后靠文案补救。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await run_search(tool, normalized_query="旅行三件套 抗造 轻便", filters=Filters(excluded_material_tags=["合成聚合物"]))
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.text)
        assert payload["hits"][0]["product_id"] == "P2120"
        assert all("合成聚合物" not in hit["material_tags"] for hit in payload["hits"])
        invoke = (await queue.get()).payload
        assert invoke["args"]["excluded_material_tags"] == ["合成聚合物"]

    async def test_product_search_tool_enforces_material_blacklist_from_context(self):
        """长期黑名单必须在工具入口兜底，不能依赖模型每次都记得传参。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
                effective_search={"parameters": {"price_max_major": None,
                    "target_currency": "CNY", "ship_to": None, "excluded_material_tags": ["合成聚合物"],
                    "required_material_tags": []}, "unverified_requirements": [], "excluded_products": []},
            ),
        )
        try:
            response = await run_search(tool, normalized_query="旅行三件套 抗造 轻便")
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.text)
        assert all("合成聚合物" not in hit["material_tags"] for hit in payload["hits"])
        invoke = (await queue.get()).payload
        assert invoke["args"]["excluded_material_tags"] == ["合成聚合物"]

    async def test_product_search_tool_does_not_infer_category_from_query(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await run_search(tool, normalized_query="户外运动 现货", filters=Filters(ship_to="CN"))
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.text)
        assert payload["hits"]
        invoke = (await queue.get()).payload
        assert invoke["args"]["category"] is None

    async def test_product_search_tool_uses_explicit_catalog_category(self):
        """工具只执行明确提交的目录分类。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="USD"),
        )
        try:
            response = await run_search(tool, normalized_query="主动降噪耳机", category="数码配件", filters=Filters(ship_to="US", target_currency="USD"))
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.text)
        assert payload["hits"]
        assert all(hit["category"] == "数码配件" for hit in payload["hits"])
        assert all("landed_price" in hit for hit in payload["hits"])
        invoke = (await queue.get()).payload
        assert invoke["args"]["category"] == "数码配件"

    async def test_product_search_tool_rejects_bad_numeric_string(self):
        """状态输入校验拒绝非法数字字符串，不把它交给查询接口。"""
        bus = TradeEventBus()
        bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            from pydantic import ValidationError
            with pytest.raises(ValidationError, match="price_max_major"):
                await run_search(tool, normalized_query="旅行三件套", filters=Filters(price_max_major="不是数字"))
        finally:
            ShoppingContext.reset(token)


    async def test_product_search_tool_rejects_unsupported_destination_deterministically(self):
        """正式故障集需要稳定注入一个不会被静默吞掉的工具错误。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await run_search(tool, normalized_query="露营灯", filters=Filters(ship_to="BR"))
        finally:
            ShoppingContext.reset(token)

        assert response.text.startswith("[error] 暂不支持的目的国")
        await queue.get()  # tool.invoke
        assert queue.empty(), "直接调用业务函数不能重复发布运行时工具结果"

    async def test_create_order_tool_and_error_path(self, confirmation_env):
        from tests.shopping_state_helpers import selected_lines
        env = confirmation_env
        tool = build_create_order_tool(PlaceOrderUseCase(env.service), env.bus, env.evidence)
        search = build_product_search_tool(CatalogSearchUseCase(env.products), env.bus, env.evidence)

        # 买家身份由 ShoppingContext 注入，而非模型入参
        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY", selected_lines=selected_lines('P1001-S1')),
        )
        try:
            # 成功路径也必须经过当前买家、当前会话的真实商品检索取证。
            await run_search(search, product_id="P1001", sku_id="P1001-S1")
            ok = await run_search(tool, 
                sku_ids=["P1001-S1"],
                shipping_address=ADDRESS,
            )
            result = json.loads(ok.text)
            assert result["confirmation_required"] is True
            assert result["confirmation"]["status"] == "pending"
            assert result["confirmation"]["buyer_id"] == "b1"
            assert "order" not in result
            assert (await env.store.get_inventory(["P1001-S1"]))["P1001-S1"] == 50

            bad = await run_search(tool, 
                sku_ids=["X"],
                shipping_address=ADDRESS,
            )
            assert bad.text.startswith("[error]")
        finally:
            ShoppingContext.reset(token)
