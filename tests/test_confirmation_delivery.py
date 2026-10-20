"""确认交付由真实业务回执收口，保留人工批准和原子交易。"""
import asyncio
from dataclasses import asdict
from langchain_core.messages import AIMessage, ToolMessage
from app.application.runtime.delivery import FinalDeliveryMiddleware

from app.application.usecases.intents import SubmitIntentInput
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_langgraph_runtime import _container
from tests.trade_test_helpers import test_address as sample_address


async def test_delivery_with_unresolved_notice_does_not_stop_the_model():
    messages = [call('cancel_order_tool', {'order_id': 'known', 'reason': 'test'}, turn_action='finish'),
        ToolMessage(content='待核查', tool_call_id='call-1', name='cancel_order_tool',
                    artifact={'execution_id': 'accepted', 'turn_action':'finish', 'notices': ['执行结果需要说明']})]
    assert await FinalDeliveryMiddleware().abefore_model({'messages': messages,
        'operation_receipts': [{'id': 'accepted', 'guidance': '请确认'}]}, None) is None
    messages[-1].artifact['notices'] = []
    assert (await FinalDeliveryMiddleware().abefore_model({'messages': messages,
        'operation_receipts': [{'id': 'accepted', 'guidance': '请确认'}]}, None))['jump_to'] == 'end'


async def test_three_turns_need_no_model_after_authoritative_confirmation(tmp_path, monkeypatch):
    model = ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1003'}),
        call('recommend_products', {'picks': [{'product_id': 'P1003', 'sku_id': 'P1003-S1',
            'quantity': 1, 'reason': '轻便', 'tradeoffs': []}], 'guidance': '这款适合随身携带。'}, turn_action='finish'),
        call('create_order_tool', {'items': [{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'quantity': 1}],
            'shipping_address': asdict(sample_address())}, turn_action='finish')])
    c = await _container(tmp_path, monkeypatch, model)
    try:
        before = await c.trade_store.get_inventory()
        first = await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '推荐背包'))
        assert first.presentations[0]['payload']['hits'][0]['skus']
        prepared = await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '选择1件，准备确认'))
        assert prepared.error is None and '确认卡' in prepared.final_text and model.cursor == 3
        cards = await c.trade_store.list_confirmations(buyer_id='b', session_id='s')
        card = cards[0]
        assert card['status'] == 'pending' and (await c.trade_store.list_orders(buyer_id='b'))['total'] == 0
        assert await c.trade_store.get_inventory() == before
        resolved = await asyncio.gather(*(c.confirmations.resolve(card['confirmation_id'], 'b', 's', card['snapshot_hash'], True) for _ in range(3)))
        assert all(x == resolved[0] for x in resolved)
        assert (await c.trade_store.list_orders(buyer_id='b'))['total'] == 1
        model.responses.append(call('cancel_order_tool', {'order_id': resolved[0]['order']['order_id'], 'reason': '重复选购'}, turn_action='finish'))
        cancelled = await c.orchestrator.handle_intent(SubmitIntentInput('s', 'b', 'zh-CN', 'CNY', '准备取消刚才订单'))
        assert cancelled.error is None and '取消确认卡' in cancelled.final_text and model.cursor == 4
        assert (await c.trade_store.get_inventory())['P1003-S1'] == before['P1003-S1'] - 1
        pending = next(x for x in await c.trade_store.list_confirmations(buyer_id='b', session_id='s') if x['action'] == 'cancel')
        one = await c.confirmations.resolve(pending['confirmation_id'], 'b', 's', pending['snapshot_hash'], True)
        again = await c.confirmations.resolve(pending['confirmation_id'], 'b', 's', pending['snapshot_hash'], True)
        assert one == again and await c.trade_store.get_inventory() == before
    finally:
        await c.shutdown()
