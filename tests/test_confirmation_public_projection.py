"""合成敏感形态标记验证原生工具审核与确认事件是否采用同一公开投影。"""
import json
from pathlib import Path
from ag_ui.core import RunAgentInput
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.delivery import FinalDeliveryMiddleware
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.order_tools import build_handle_order_tool
from app.application.usecases.order_usecases import OrderItemInput,HandleOrderUseCase
from app.infrastructure.eventbus import observe_run_events
from tests.test_agent_handoff import ScriptedModel,call
from tests.trade_test_helpers import confirmation_env,test_address as address

async def test_confirmation_and_tool_share_public_projection(confirmation_env):
    e=confirmation_env
    prepared=await e.service.prepare_order('audit-buyer','audit-session',[OrderItemInput('P1003','P1003-S1',1)],address(),'CNY')
    c=prepared['confirmation']
    approved=await e.service.resolve(c['confirmation_id'],'audit-buyer','audit-session',c['snapshot_hash'],True)
    marker='sk-'+'SyntheticAuditMarker1234567890'
    model=ScriptedModel(responses=[call('handle_order',{'order_id': approved['order']['order_id'], 'cancellation': {'reason': marker}},turn_action='finish')])
    tool=as_langchain_tool(build_handle_order_tool(HandleOrderUseCase(e.service),e.bus),turn_delivery=True)
    graph=create_agent(model,tools=[tool],middleware=[BusinessToolMiddleware(None,e.bus),FinalDeliveryMiddleware()])
    body=RunAgentInput.model_validate({'threadId':'audit-session','runId':'audit-run','messages':[{'id':'u','role':'user','content':'取消测试订单'}],
        'state':{},'tools':[],'context':[],'forwardedProps':{'buyerId':'audit-buyer'}})
    frames=[];events=[];adapter=AGUIRunAdapter(body,frames.append)
    token=ShoppingContext.set(ShoppingContextSnapshot('audit-session','audit-buyer','zh-CN','CNY',source_run_id='audit-run'))
    def capture(event):events.append(event);adapter.on_trade_event(event)
    try:
        with observe_run_events(capture):
            await graph.ainvoke({'messages':[HumanMessage(name='audit-buyer',content='取消测试订单')]})
        raw=next(event for event in events if event.type=='confirmation.required')
        normalized=next(event for event in events if event.type=='tool.result')
        assert marker not in json.dumps(raw.to_dict())
        assert marker not in json.dumps(normalized.to_dict())
        assert marker not in json.dumps(adapter.state)
        assert all(marker not in frame.model_dump_json() for frame in frames)
        card=raw.payload['confirmation']
        assert card['payload']['reason']==marker
        saved=await e.service.get(card['confirmation_id'],'audit-buyer','audit-session')
        assert saved['confirmation']['payload']['reason']==marker
        assert raw.to_dict()['payload']['confirmation']['snapshot_hash']==card['snapshot_hash']
    finally:ShoppingContext.reset(token)

import pytest
from app.infrastructure.security.public_projection import PublicProjection
from app.infrastructure.ag_ui_journal import AGUIJournal
from app.presentation.confirmations import register_confirmation_routes
from fastapi import FastAPI
import httpx

@pytest.mark.parametrize('enabled',[True,False])
async def test_confirmation_http_and_persisted_restore_share_policy(confirmation_env,tmp_path,enabled):
    e=confirmation_env;policy=PublicProjection(enabled);e.bus.projection=policy
    marker='sk-'+'SyntheticAuditMarker1234567890'
    prepared=await e.service.prepare_order('buyer','session',[OrderItemInput('P1003','P1003-S1',1)],address(),'CNY')
    card=prepared['confirmation']
    order=(await e.service.resolve(card['confirmation_id'],'buyer','session',card['snapshot_hash'],True))['order']
    cancel=(await e.service.handle_order('buyer', 'session', order['order_id'], {'reason': marker}))['confirmation']
    app=FastAPI();register_confirmation_routes(app,lambda:e.service,projection=lambda:policy)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        for path in ['/commerce/confirmations','/commerce/confirmations/'+cancel['confirmation_id']]:
            response=await client.get(path,params={'buyer_id':'buyer','session_id':'session'})
            assert response.status_code==200 and (marker not in response.text)==enabled
        response=await client.post('/commerce/confirmations/'+cancel['confirmation_id']+'/resolve',json={
            'buyer_id':'buyer','session_id':'session','snapshot_hash':cancel['snapshot_hash'],'approved':True})
        assert response.status_code==200 and (marker not in response.text)==enabled
        assert response.json()['confirmation']['snapshot_hash']==cancel['snapshot_hash']
    journal=AGUIJournal(tmp_path/'public-journal.db',projection=policy)
    body={'threadId':'session','runId':'run','messages':[{'id':'u','role':'user','content':marker}]}
    run,_=await journal.reserve(body,'buyer','owner')
    # 原始历史仅存一次，开启策略后的导出也必须覆盖先前未投影的记录。
    await journal.append('run','owner',[{'type':'STATE_SNAPSHOT','snapshot':{'confirmations':[cancel]}},
        {'type':'MESSAGES_SNAPSHOT','messages':[*run['messages'],{'id':'a','role':'assistant','content':marker}]},
        {'type':'RUN_FINISHED','threadId':'session','runId':'run'}])
    events,_,_=await journal.events('run','buyer')
    assert (marker not in json.dumps(events[0]))==enabled
    session=await journal.session('session','buyer')
    assert session['messages'][0]['content']==marker
    assert (marker not in session['messages'][1]['content'])==enabled
    assert (marker not in json.dumps(session['state']))==enabled
    raw=await journal.run('run','buyer')
    assert raw['state']['confirmations'][0]['payload']['reason']==marker
    assert (await e.service.get(cancel['confirmation_id'],'buyer','session'))['confirmation']['payload']['reason']==marker


def test_display_projection_does_not_rewrite_protocol_fields():
    marker='sk-'+'SyntheticAuditMarker1234567890';policy=PublicProjection()
    raw={'product_id':marker,'sku_ids':[marker],'tool':'product_search_tool','snapshot_hash':marker,
         'guidance':marker,'items':[{'title':marker,'quantity':2,'unit_price_minor':123}]}
    public=policy.business(raw)
    assert public['product_id']==public['sku_ids'][0]==public['snapshot_hash']==marker
    assert public['tool']=='product_search_tool' and marker not in public['guidance']
    assert public['items'][0]['unit_price_minor']==123 and raw['guidance']==marker
