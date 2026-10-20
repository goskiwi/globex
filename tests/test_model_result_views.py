"""完整卡片和模型回执各走自己的数据边界。"""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, ToolMessage

from app.application.runtime.model_views import product_result_view
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


async def test_delivered_card_stays_complete_while_next_model_receives_only_receipt(tmp_path, monkeypatch):
    model = ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1003'}),
        AIMessage(content='', tool_calls=[
            {'id': 'present', 'name': 'recommend_products', 'args': {
                'guidance': '这款适合随身携带，详细资料在卡片。',
                'picks': [{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'quantity': 1,
                           'reason': '重量轻且可折叠', 'tradeoffs': ['未标注电脑隔层']}]}},
            {'id': 'read', 'name': 'query_order_tool', 'args': {'order_id': 'UNKNOWN'}}]),
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
