"""完整卡片和模型回执各走自己的数据边界。"""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, ToolMessage

from app.application.runtime.model_views import product_result_view, historical_products, historical_product_views
from app.application.usecases.intents import SubmitIntentInput
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_langgraph_runtime import _container


def test_only_identical_default_quote_is_shared_and_other_sku_quote_survives():
    quote = {'ship_to': 'CN', 'currency': 'CNY', 'total_amount_minor': 1500}
    raw = {'hits': [{'product_id': 'P1003', 'default_sku_id': 'P1003-S1',
        'description': '有电脑隔层，重量0.68kg', 'weight_kg': .68,
        'landed_price': quote, 'skus': [
            {'sku_id': 'P1003-S1', 'landed_price': deepcopy(quote)},
            {'sku_id': 'P1003-S2', 'landed_price': {**quote, 'ship_to': 'JP'}}]}]}
    saved = deepcopy(raw)
    hit = product_result_view(raw)['hits'][0]
    assert 'landed_price' not in hit['skus'][0]
    assert hit['landed_price'] == quote and hit['skus'][1]['landed_price']['ship_to'] == 'JP'
    assert hit['weight_kg'] == .68 and hit['description'] == raw['hits'][0]['description']
    assert raw == saved


def test_history_retains_only_selected_sku_facts_and_keeps_recoverable_candidate_order():
    raw = {'result_ref': 'saved', 'observed_at': '2026-10-20', 'hits': [
        {'product_id': 'P1003', 'title': '背包', 'default_sku_id': 'P1003-S1',
         'price_major': 99, 'landed_price': {'currency': 'CNY', 'total_amount_minor': 9900},
         'weight_kg': .68, 'description': '保留实际规格事实',
         'skus': [{'sku_id': 'P1003-S1'}, {'sku_id': 'P1003-S2', 'price_major': 20, 'currency': 'USD'}]},
        {'product_id': 'P1049', 'title': '另一个背包', 'skus': [{'sku_id': 'P1049-S1'}]}]}
    view = historical_products(raw, {'P1003', 'P1003-S2'})
    assert len(view['hits']) == 1 and view['hits'][0]['skus'] == raw['hits'][0]['skus'][1:]
    assert view['hits'][0]['weight_kg'] == .68 and 'landed_price' not in view['hits'][0]
    assert [x['product_id'] for x in view['candidate_index']] == ['P1003', 'P1049']
    assert view['result_ref'] == 'saved' and view['historical']
    assert len(raw['hits']) == 2 and len(raw['hits'][0]['skus']) == 2


def test_repeated_selected_sku_keeps_latest_observation_without_pinning_old_history():
    old = {'result_ref': 'old', 'hits': [{'product_id': 'P1003', 'title': '背包',
        'default_sku_id': 'P1003-S1', 'skus': [{'sku_id': 'P1003-S1', 'stock': 5}]}]}
    new = deepcopy(old)
    new['result_ref'] = 'new'
    new['hits'][0]['skus'][0]['stock'] = 3
    views = historical_product_views([('old-call', old), ('new-call', new)], {'P1003-S1'})
    assert views['old-call']['archived'] and not views['old-call']['hits']
    assert views['old-call']['result_ref'] == 'old'
    assert views['new-call']['hits'][0]['skus'][0]['stock'] == 3
    assert old['hits'][0]['skus'][0]['stock'] == 5


def test_historical_query_currencies_remain_separate_when_no_landed_quote_exists():
    first = {'result_ref': 'cny', 'query_conditions': {'target_currency': 'CNY'}, 'hits': [
        {'product_id': 'P1003', 'title': '背包', 'skus': [{'sku_id': 'P1003-S1'}]}]}
    second = deepcopy(first)
    second['result_ref'] = 'usd'
    second['query_conditions']['target_currency'] = 'USD'
    views = historical_product_views([('a', first), ('b', second)], {'P1003-S1'})
    assert views['a']['hits'] and views['b']['hits']


async def test_delivered_card_stays_complete_while_next_model_receives_only_receipt(tmp_path, monkeypatch):
    model = ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1003'}),
        AIMessage(content='', tool_calls=[
            {'id': 'present', 'name': 'recommend_products', 'args': {
                    'guidance': '这款适合随身携带，详细资料在卡片。', 'turn_action':'finish',
                'picks': [{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'quantity': 1,
                           'reason': '重量轻且可折叠', 'tradeoffs': ['未标注电脑隔层']}]}},
            {'id': 'read', 'name': 'handle_order', 'args': {'order_id': 'UNKNOWN', 'turn_action': 'continue'}}]),
        AIMessage(content='卡片已展示，订单号未找到。')])
    container = await _container(tmp_path, monkeypatch, model)
    catalog = container.orchestrator._sessions._main_factory._search_factory._catalog_search
    catalog._embedder = catalog._vector_index = catalog._reranker = None
    try:
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '推荐背包，同时查询订单UNKNOWN'))
        card = result.presentations[0]['payload']['hits'][0]
        assert card['title'] and card['skus'] and card['description']
        assert card['recommendation_reason'] == '重量轻且可折叠'
        receipt = next(m for m in model.seen[-1] if isinstance(m, ToolMessage) and m.name == 'recommend_products')
        data = json.loads(receipt.content)
        assert data['displayed'] and data['sku_ids'] == ['P1003-S1'] and 'hits' not in data
    finally:
        await container.shutdown()
