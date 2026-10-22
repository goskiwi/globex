"""完整订单任务：同一事务视图、业务拒绝、归属和确认后独立核验。"""
import pytest
from app.application.usecases.order_usecases import OrderItemInput,HandleOrderUseCase
from tests.trade_test_helpers import confirmation_env,test_address as address

async def create(e, session='session'):
    p=await e.service.prepare_order('buyer',session,[OrderItemInput('P1003','P1003-S1',1)],address(),'CNY')
    c=p['confirmation'];return (await e.service.resolve(c['confirmation_id'],'buyer',session,c['snapshot_hash'],True))['order']

@pytest.mark.parametrize('cancel',[False,True])
async def test_order_task_reads_once_and_returns_complete_order(confirmation_env,cancel):
    e=confirmation_env;order=await create(e);inventory=await e.store.get_inventory()
    confirmations=await e.store.list_confirmations(buyer_id='buyer',session_id='session')
    reads=[];original=e.store._owned_order
    async def observed(*args):reads.append(args[1]);return await original(*args)
    e.store._owned_order=observed
    outcome=await HandleOrderUseCase(e.service).execute(order['order_id'],buyer_id='buyer',session_id='session',
        cancellation={'reason':'重复选购'} if cancel else None)
    assert reads==[order['order_id']] and outcome['order']==order
    assert outcome['order']['created_at'] and outcome['order']['status']=='CONFIRMED'
    assert outcome['confirmation_required'] is cancel and outcome['status']=='completed'
    assert await e.store.get_inventory()==inventory
    if cancel:
        card=outcome['confirmation'];assert card['payload']['order_status']==outcome['order']['status']
        assert card['payload']['total_amount_minor']==outcome['order']['total_amount_minor']
        assert outcome['operation']['status']=='awaiting_confirmation'
        reads.clear()
        await e.service.resolve(card['confirmation_id'],'buyer','session',card['snapshot_hash'],True)
        assert reads==[order['order_id']], '买家确认是另一次独立权威核验，不能用准备快照替代'
    else:
        assert outcome['confirmation'] is None and outcome['operation']['kind']=='inspect'
        assert await e.store.list_confirmations(buyer_id='buyer',session_id='session')==confirmations
        assert (await e.store.list_orders(buyer_id='buyer'))['total']==1

async def test_rejected_cancellation_keeps_owned_current_order_without_confirmation(confirmation_env):
    e=confirmation_env;order=await create(e)
    first=await e.service.handle_order('buyer','session',order['order_id'],{'reason':'重复选购'})
    card=first['confirmation'];await e.service.resolve(card['confirmation_id'],'buyer','session',card['snapshot_hash'],True)
    before=await e.store.list_confirmations(buyer_id='buyer',session_id='session')
    result=await e.service.handle_order('buyer','session',order['order_id'],{'reason':'再次取消'})
    assert result['order']['status']=='CANCELLED' and result['status']=='partial'
    assert result['operation']['rejection']['code']=='ORDER_CHANGED'
    assert result['confirmation'] is None and not result['confirmation_required']
    assert await e.store.list_confirmations(buyer_id='buyer',session_id='session')==before

@pytest.mark.parametrize('cancel',[False,True])
async def test_unowned_order_never_returns_a_view(confirmation_env,cancel):
    e=confirmation_env;order=await create(e)
    with pytest.raises(ValueError):
        await e.service.handle_order('other','other-session',order['order_id'],{'reason':'越权'} if cancel else None)

@pytest.mark.parametrize('mode',['inspect','cancel','rejected'])
async def test_model_order_task_delivers_once_and_restores_across_turns(tmp_path,monkeypatch,mode):
    from types import SimpleNamespace
    from tests.test_langgraph_runtime import _container
    from tests.test_agent_handoff import ScriptedModel,call
    from tests.test_execution_facts import request
    from scripts.eval.trace import tool_observations
    model=ScriptedModel(responses=[])
    c=await _container(tmp_path,monkeypatch,model)
    e=SimpleNamespace(service=c.confirmations,store=c.trade_store)
    try:
        order=await create(e,session='facts-session')
        if mode=='rejected':
            p=await c.confirmations.handle_order('buyer','facts-session',order['order_id'],{'reason':'已取消'})
            card=p['confirmation'];await c.confirmations.resolve(card['confirmation_id'],'buyer','facts-session',card['snapshot_hash'],True)
        reads=[];original=c.trade_store._owned_order
        async def measured(*args):reads.append(args[1]);return await original(*args)
        c.trade_store._owned_order=measured
        args={'order_id':order['order_id']}
        if mode!='inspect':args['cancellation']={'reason':'重复选购'}
        model.responses.append(call('handle_order',args,turn_action='finish'))
        await c.runs.startup();await c.runs.start(request('order-task'),'buyer')
        saved=await c.runs.wait('order-task','buyer',10)
        assert saved['result']['status']==('partial' if mode=='rejected' else 'completed')
        assert model.cursor==1
        assert reads==[order['order_id']], '模型前上下文只给引用，当前订单在任务内读取一次'
        panels=saved['state']['presentations'];assert len(panels)==1 and panels[0]['kind']=='order'
        data=panels[0]['payload'];assert data['order']['created_at'] and data['observed_at']
        assert data['operation']['status']=={'inspect':'completed','cancel':'awaiting_confirmation','rejected':'rejected'}[mode]
        rows,_,_=await c.runs.journal.events('order-task','buyer')
        assert [r['name'] for r in tool_observations([r['event'] for r in rows])]==['handle_order']
        assert saved['messages'][-1]['content']==data['guidance']
        history=await c.runs.journal.session('facts-session','buyer')
        assert history['presentations']==panels
        tools=c.orchestrator._sessions._agents['facts-session'].tool_names
        assert 'handle_order' in tools and not {'query_order_tool','cancel_order_tool'}&tools
    finally:await c.shutdown()
