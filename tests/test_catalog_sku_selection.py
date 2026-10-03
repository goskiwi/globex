"""以真实 SKU 为资格单位：检索、证据、展示及确认单保持同一对象。"""
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.order_usecases import PlaceOrderUseCase
from app.application.tools.order_tools import build_create_order_tool
from app.application.tools.recommendation_tools import build_recommendation_tool, Pick
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.sku import Sku
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.persistence.seed_products import build_seed_products
from tests.trade_test_helpers import confirmation_env, test_address as address


def product():
    return Product(product_id="P8800", title="旅行背包", brand="测试", category="旅行装备",
        origin_country="CN", description="随身旅行收纳", ships_to=["CN"], skus=[
            Sku("P8800-S1", "黑色 / M", Money.from_major_units(180, "CNY"), 5),
            Sku("P8800-S2", "沙漠黄 / XL", Money.from_major_units(80, "CNY"), 5),
            Sku("P8800-S3", "红色 / S", Money.from_major_units(10, "CNY"), 0),
            Sku("P8800-S4", "蓝色 / L", Money.from_major_units(40, "CNY"), 5)])


def pipeline(item, mode):
    index = SimpleNamespace(
        search=AsyncMock(return_value=[VectorHit(item.product_id, 1.0)]),
        search_filtered=AsyncMock(side_effect=lambda vector, top_n, product_ids:
            [VectorHit(item.product_id, 1.0)] if item.product_id in product_ids else []))
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=lambda query, documents: [1.0] * len(documents)))
    return CatalogSearchUseCase(InMemoryProductRepository([item]),
        embedder=SimpleNamespace(embed=AsyncMock(return_value=[1.0])) if "vector" in mode else None,
        vector_index=index if "vector" in mode else None,
        reranker=reranker, hybrid_enabled=mode.startswith("hybrid")), reranker


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid", "hybrid_vector", "exact_product"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_all_search_paths_select_eligible_skus_independent_of_catalog_order(mode, reverse):
    item = product()
    if reverse:
        item.skus.reverse()
    search, reranker = pipeline(item, mode)
    spec = ProductSearchSpec(normalized_query="旅行背包", price_max_major=100, ship_to="CN",
        excluded_sku_ids=("P8800-S4",), product_id=item.product_id if mode == "exact_product" else None)
    result = await search.execute(spec)
    assert len(result["hits"]) == 1
    card = result["hits"][0]
    assert card["default_sku_id"] == "P8800-S2" and card["price_major"] == 80
    assert [sku["sku_id"] for sku in card["skus"]] == ["P8800-S2"]
    assert card["landed_price"]["items"][0]["sku_id"] == "P8800-S2"
    assert card["landed_price"]["subtotal_minor"] == 8000
    if reranker.rerank.await_count:
        text = reranker.rerank.await_args.args[1][0]
        assert "沙漠黄 / XL" in text
        assert all(spec not in text for spec in ("黑色 / M", "红色 / S", "蓝色 / L"))


async def test_explicit_sku_is_never_replaced_by_affordable_sibling():
    search, _ = pipeline(product(), "hybrid_vector")
    result = await search.execute(ProductSearchSpec(sku_id="P8800-S1", price_max_major=100))
    assert result["hits"] == []
    rejected = result["filtered_out"][0]
    assert rejected["reason"] == "over_price_cap"
    assert rejected["skus"] == [{"sku_id": "P8800-S1", "spec": "黑色 / M",
        "price_major": 180, "currency": "CNY", "reasons": ["over_price_cap"]}]


async def test_price_stock_and_exclusion_cannot_come_from_different_skus():
    search, _ = pipeline(product(), "hybrid_vector")
    result = await search.execute(ProductSearchSpec(product_id="P8800", price_max_major=50,
        excluded_sku_ids=("P8800-S4",)))
    assert not result["hits"]
    rows = {row["sku_id"]: row for row in result["filtered_out"][0]["skus"]}
    assert rows["P8800-S1"]["reasons"] == ["over_price_cap"]
    assert rows["P8800-S3"]["reasons"] == ["out_of_stock"]
    assert rows["P8800-S4"]["reasons"] == ["sku_excluded"]


async def test_default_price_compares_converted_prices_with_deterministic_ties():
    item = product()
    item.skus = [Sku("P8800-S2", "美元款", Money.from_major_units(10, "USD"), 2),
                 Sku("P8800-S1", "人民币款", Money.from_major_units(50, "CNY"), 2)]
    search, _ = pipeline(item, "keyword")
    card = (await search.execute(ProductSearchSpec(product_id=item.product_id, price_max_major=60)))["hits"][0]
    assert card["default_sku_id"] == "P8800-S1" and card["price_major"] == 50
    item.skus[0].price = Money.from_major_units(50, "CNY")
    for _ in range(2):
        item.skus.reverse()
        card = (await search.execute(ProductSearchSpec(product_id=item.product_id)))["hits"][0]
        assert card["default_sku_id"] == "P8800-S1"


@pytest.mark.parametrize("hybrid", [False, True])
async def test_real_catalog_color_in_sku_is_searchable(hybrid):
    item = next(p for p in build_seed_products() if p.product_id == "P1001")
    search = CatalogSearchUseCase(InMemoryProductRepository([item]), hybrid_enabled=hybrid)
    result = await search.execute(ProductSearchSpec(normalized_query="沙漠黄"))
    assert [p["product_id"] for p in result["hits"]] == ["P1001"]
    assert any(s["spec"] == "沙漠黄" for s in result["hits"][0]["skus"])


async def test_search_recommendation_details_and_confirmation_use_the_same_sku(confirmation_env):
    e = confirmation_env
    catalog = CatalogSearchUseCase(e.products, pricing=e.service._pricing)
    spec = ProductSearchSpec(product_id="P2101", price_max_major=100, target_currency="USD", ship_to="EU")
    result = await catalog.execute(spec)
    assert result["hits"][0]["default_sku_id"] == "P2101-S2"
    assert result["hits"][0]["price_major"] == 6.15
    await e.evidence.save("sku-buyer", "sku-session", "products", result)
    policy = {"parameters": {"price_max_major": 100, "target_currency": "USD", "ship_to": "EU"}}
    context = ShoppingContextSnapshot("sku-session", "sku-buyer", "zh-CN", "USD",
        effective_search=policy, selected_lines=({"product_id": "P2101", "sku_id": "P2101-S2", "quantity": 1},))
    token = ShoppingContext.set(context)
    try:
        recommend = build_recommendation_tool(catalog, e.evidence, e.bus)
        pick = Pick(product_id="P2101", sku_id="P2101-S2", quantity=1, reason="此规格在预算内")
        reply = await recommend([pick], "alternatives", pick.sku_id, [], "可选择这款餐盒。")
        assert reply.ok
        card = json.loads(reply.text)["hits"][0]
        assert [s["sku_id"] for s in card["skus"]] == [pick.sku_id]
        details = await catalog.product_details("P2101", spec)
        rows = {s["sku_id"]: s for s in details["hits"][0]["skus"]}
        assert "over_price_cap" in rows["P2101-S1"]["constraint_issues"]
        assert rows[pick.sku_id]["constraint_issues"] == []
        create = build_create_order_tool(PlaceOrderUseCase(e.service), e.bus, e.evidence)
        before = await e.store.get_inventory()
        prepared = await create([pick.sku_id], asdict(replace(address(), country="EU")))
        assert prepared.ok
        confirmation = json.loads(prepared.text)["confirmation"]
        assert confirmation["payload"]["items"][0]["sku_id"] == pick.sku_id
        assert confirmation["payload"]["items"][0]["unit_price_minor"] == 615
        assert await e.store.get_inventory() == before
        changed = {"parameters": {**policy["parameters"], "price_max_major": 5}}
        ShoppingContext.set(replace(context, effective_search=changed))
        assert not (await recommend([pick], "alternatives", pick.sku_id, [], "再核对预算。 ")).ok
        blocked = await catalog.product_details("P2101", replace(spec, price_max_major=5))
        assert all("over_price_cap" in s["constraint_issues"] for s in blocked["hits"][0]["skus"])
    finally:
        ShoppingContext.reset(token)
