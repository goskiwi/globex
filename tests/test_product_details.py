"""完整目录读取属于详情卡内部；推荐资料只来自检索，展示不授予交易资格。"""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
from app.application.tools.product_view_tool import ProductTarget, build_product_view_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.recommendation_tools import build_recommendation_tool, Pick
from app.application.tools.order_tools import _verified_order_items
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env


def test_view_target_has_no_budget_override_and_rejects_mismatched_sku():
    assert ProductTarget(product_id='P3020').sku_id is None
    with pytest.raises(ValidationError):
        ProductTarget(product_id='P3020', price_max_major=None)
    with pytest.raises(ValidationError):
        ProductTarget(product_id='P3020', sku_id='P3021-S1')


async def test_complete_blocked_details_are_displayed_without_changing_filters(confirmation_env):
    e = confirmation_env
    catalog = CatalogSearchUseCase(e.products, pricing=e.service._pricing)
    work = ShoppingWork(filters=Filters(price_max_major=100, target_currency='CNY'), excluded_skus=['P3021-S1'])
    effective = compile_search(work)
    before = deepcopy(effective)
    token = ShoppingContext.set(ShoppingContextSnapshot('detail-test', 'buyer', 'zh-CN', 'CNY', effective_search=effective))
    try:
        show = build_product_view_tool(catalog, e.evidence, e.bus)
        search = build_product_search_tool(catalog, e.bus, e.evidence)
        for product_id, platform in [('P3020', 'amazon'), ('P3021', 'ebay')]:
            returned = await show([{'product_id': product_id}], '查看颈枕的完整资料。')
            assert returned.ok
            p = returned.data['hits'][0]
            assert p['source_platform'] == platform and len(p['skus']) == 2
            assert p['description'] and p['dimensions_cm'] and p['weight_kg'] == .15
            assert all('over_price_cap' in s['constraint_issues'] for s in p['skus'])
            if product_id == 'P3020':
                assert all(s['stock'] == 0 and 'out_of_stock' in s['constraint_issues'] for s in p['skus'])
            else:
                assert p['skus'][1]['display_price_major'] == 213.99
                assert p['skus'][1]['currency'] == 'USD' and p['skus'][1]['stock'] > 0
                assert p['skus'][0]['sku_id'] == 'P3021-S1'
            blocked = (await search(product_id=product_id)).data
            assert blocked['hits'] == [] and blocked['filtered_out']
        assert effective == before and work.filters.price_max_major == 100
        assert await e.evidence.find_product('buyer', 'detail-test', product_id='P3021', sku_id='P3021-S2') is None
        with pytest.raises(ValueError):
            await _verified_order_items(e.evidence, 'buyer', 'detail-test', [{'product_id': 'P3021', 'sku_id': 'P3021-S2', 'quantity': 1}])
        refused = await build_recommendation_tool(catalog, e.evidence, e.bus)([Pick(product_id='P3021', sku_id='P3021-S2', quantity=1)], [])
        assert not refused.ok
        assert len(await e.evidence.search('buyer', 'detail-test', kind='product_view')) == 2
    finally:
        ShoppingContext.reset(token)


async def test_catalog_exact_missing_ids_and_all_skus_even_when_specific_sku_requested(confirmation_env):
    catalog = CatalogSearchUseCase(confirmation_env.products)
    spec = ProductSearchSpec(normalized_query='颈枕', target_currency='CNY')
    for product_id, sku_id, missing in [('P99999', None, 'P99999'), ('P3020', 'P3020-S99', 'P3020-S99')]:
        data = await catalog.product_details(product_id, spec, sku_id=sku_id)
        assert data['hits'] == [] and data['existence_checked'] and data['missing_identifiers'] == [missing]
    p = (await catalog.product_details('P3020', spec, sku_id='P3020-S2'))['hits'][0]
    assert p['default_sku_id'] == 'P3020-S2' and len(p['skus']) == 2


async def test_readonly_quote_uses_display_without_granting_candidate_permission(confirmation_env):
    from app.application.tools.recommendation_tools import build_quote_tool, Choice
    e = confirmation_env
    session = 'detail-quote'
    catalog = CatalogSearchUseCase(e.products, pricing=e.service._pricing)
    token = ShoppingContext.set(ShoppingContextSnapshot(session, 'buyer', 'zh-CN', 'CNY',
        effective_search=compile_search(ShoppingWork(filters=Filters(price_max_major=1, target_currency='CNY')))))
    try:
        shown = await build_product_view_tool(catalog, e.evidence, e.bus)(
            [{'product_id': 'P3021', 'sku_id': 'P3021-S2'}], '查看颈枕的当前资料。')
        assert shown.ok
        quote = await build_quote_tool(catalog.pricing, e.evidence, e.bus)(
            [Choice(product_id='P3021', sku_id='P3021-S2', quantity=1)], 'CN', 'CNY')
        assert quote.ok and quote.data['quote']['items'][0]['sku_id'] == 'P3021-S2'
        assert await e.evidence.find_product('buyer', session, product_id='P3021', sku_id='P3021-S2') is None
        with pytest.raises(ValueError):
            await _verified_order_items(e.evidence, 'buyer', session, [{'product_id': 'P3021', 'sku_id': 'P3021-S2', 'quantity': 1}])
        assert await e.evidence.find_observed_product('other', session, product_id='P3021', sku_id='P3021-S2') is None
        assert (await e.store.list_orders(buyer_id='buyer'))['total'] == 0
    finally:
        ShoppingContext.reset(token)


async def test_native_search_supplies_recommendation_facts_without_another_catalog_read(tmp_path, monkeypatch):
    import json
    from langchain_core.messages import ToolMessage
    from tests.test_agent_handoff import ScriptedModel, call
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput

    model = ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1003', 'conditions': {'filters': {'price_max_major': 200, 'target_currency': 'CNY', 'ship_to': 'CN'}}}), call('recommend_products', {'picks': [{'product_id': 'P1003', 'sku_id': 'P1003-S2', 'quantity': 1}], 'focus': []}, turn_action='finish')])
    container = await _container(tmp_path, monkeypatch, model)
    try:
        result = await container.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY',
            '推荐200元以内、寄中国的通勤背包，展示商品卡，先不下单。'))
        assert result.status == 'completed' and model.cursor == 2
        session = container.orchestrator._sessions._agents['s']
        tools = {tool.name for tool in session.context_policy.tools}
        assert 'get_product_details' not in tools
        receipt = next(m for m in model.seen[1] if isinstance(m, ToolMessage) and m.name == 'product_search_tool')
        product = json.loads(receipt.content)['hits'][0]
        assert product['description'] and product['weight_kg'] == .38
        assert product['rating_summary']['average'] == 4.2
        assert {sku['sku_id'] for sku in product['skus']} == {'P1003-S1', 'P1003-S2'}
        assert all(sku['stock'] > 0 for sku in product['skus'])
        card = result.presentations[0]['payload']['hits'][0]
        assert card['default_sku_id'] == 'P1003-S2' and card['weight_kg'] == product['weight_kg']
        assert card['landed_price']['items'][0]['sku_id'] == 'P1003-S2'
        assert (await container.trade_store.list_orders(buyer_id='b'))['total'] == 0
    finally:
        await container.shutdown()
