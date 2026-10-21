"""只审查：临时SQLite确认流程中实际订单读取次数，不访问生产账本。"""
import asyncio
import json
from datetime import timedelta
from pathlib import Path
from app.application.usecases.order_usecases import OrderItemInput, QueryOrderUseCase, CancelOrderUseCase
from app.application.tools.order_tools import build_query_order_tool, build_cancel_order_tool
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env, test_address as address

async def test_cancel_order_reads(confirmation_env):
    e=confirmation_env
    prepared=await e.service.prepare_order('synthetic-buyer','synthetic-session',
        [OrderItemInput('P1003','P1003-S1',1)],address(),'CNY')
    c=prepared['confirmation']
    resolved=await e.service.resolve(c['confirmation_id'],'synthetic-buyer','synthetic-session',c['snapshot_hash'],True)
    order=resolved['order'];inventory=await e.store.get_inventory()
    owned=[]
    original=e.store._owned_order
    async def observed(db,order_id,buyer_id):
        owned.append((order_id,buyer_id))
        return await original(db,order_id,buyer_id)
    e.store._owned_order=observed
    token=ShoppingContext.set(ShoppingContextSnapshot('synthetic-session','synthetic-buyer','zh-CN','CNY'))
    query=build_query_order_tool(QueryOrderUseCase(e.store),e.bus)
    cancel=build_cancel_order_tool(CancelOrderUseCase(e.service),e.bus)
    samples=[]
    try:
        for with_query in [False,True]:
            owned.clear()
            results=await asyncio.gather(*([query(order['order_id'])] if with_query else []),cancel(order['order_id'],'合成测试'))
            assert all(r.ok for r in results)
            count=len(owned)
            samples.append({'parallel_query_tool':with_query,'owned_order_reads':count})
            assert count==(2 if with_query else 1)
        # 验证权威准备事务自身也校验归属，不依赖外层get_order的预读。
        try:
            await e.store.prepare_confirmation(operation_id='audit-foreign',buyer_id='other-buyer',session_id='other-session',
                action='cancel',payload={'order_id':order['order_id'],'reason':'合成测试'},expires_at=e.now[0]+timedelta(minutes=5))
        except ValueError as error:
            assert type(error).__name__=='TradeStoreError'
        else:
            raise AssertionError('跨买家准备不应成功')
        assert await e.store.get_inventory()==inventory
    finally:
        ShoppingContext.reset(token)
