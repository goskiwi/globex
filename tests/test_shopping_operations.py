"""业务输入直接提交，状态归属和真实确认校验仍在服务端。"""
from dataclasses import asdict
import json

from langchain_core.messages import AIMessage, ToolMessage

from app.application.usecases.intents import SubmitIntentInput
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_langgraph_runtime import _container
from tests.trade_test_helpers import test_address as sample_address


async def test_search_applies_current_conditions_without_a_separate_state_round(tmp_path, monkeypatch):
    model = ScriptedModel(responses=[call('product_search_tool', {
        'product_id': 'P1003', 'conditions': {'filters': {'price_max_major': 200,
        'target_currency': 'CNY', 'ship_to': 'CN'}}}), AIMessage(content='已查询'),
        call('product_search_tool', {'product_id': 'P1003',
            'conditions': {'filters': {'price_max_major': None}}}), AIMessage(content='已取消价格上限')])
    c = await _container(tmp_path, monkeypatch, model)
    try:
        first = await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '背包200元，寄中国'))
        assert first.error is None and model.cursor == 2
        state = (await c.orchestrator._sessions.read_checkpoint('s')).checkpoint['channel_values']
        receipt = next(m for m in state['messages'] if isinstance(m, ToolMessage) and m.name == 'product_search_tool')
        assert receipt.artifact['data']['query_conditions']['price_max_major'] == 200
        assert state['shopping_work']['filters']['ship_to'] == 'CN'
        await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '不限价格，其他不变'))
        state = (await c.orchestrator._sessions.read_checkpoint('s')).checkpoint['channel_values']
        assert state['shopping_work']['filters']['price_max_major'] is None
        assert state['shopping_work']['filters']['ship_to'] == 'CN'
    finally:
        await c.shutdown()


async def test_order_preparation_accepts_explicit_items_and_saves_them_without_select_round(tmp_path, monkeypatch):
    model = ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1003'}), AIMessage(content='已查询'),
        call('create_order_tool', {'items': [{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'quantity': 2}],
                                 'shipping_address': asdict(sample_address())}), AIMessage(content='等待确认')])
    c = await _container(tmp_path, monkeypatch, model)
    try:
        await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '查询这个背包'))
        before = await c.trade_store.get_inventory()
        result = await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '选择两件，准备确认'))
        assert result.error is None
        state = (await c.orchestrator._sessions.read_checkpoint('s')).checkpoint['channel_values']
        assert state['shopping_work']['selections']['P1003-S1']['quantity'] == 2
        cards = await c.trade_store.list_confirmations(buyer_id='b', session_id='s')
        assert len(cards) == 1 and cards[0]['status'] == 'pending'
        assert cards[0]['payload']['items'][0]['quantity'] == 2
        assert (await c.trade_store.list_orders(buyer_id='b'))['total'] == 0
        assert await c.trade_store.get_inventory() == before
    finally:
        await c.shutdown()
