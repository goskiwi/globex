"""选购链路：可信输入、已提交表单和未核算的到手预算，不要求模型复制原文。"""
import asyncio
import json
from dataclasses import replace

import pytest
from ag_ui.core import RunAgentInput
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.application.agents.shopping_state import ShoppingWork, ShoppingUpdate, Filters, apply_update
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.working_state import WorkingStateMiddleware
from app.application.tools.shopping_state_tool import build_shopping_state_tool
from app.application.tools.recommendation_tools import build_recommendation_tool, Pick
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.shopping_forms import ShoppingFormStore
from app.infrastructure.ag_ui_journal import AGUIJournal, JournalConflict
from app.presentation.ag_ui import parse_intent
from app.presentation.ag_ui_runtime import AGUIRuntime
from tests.test_agent_handoff import ScriptedModel
from tests.test_ag_ui_journal import body, ControlledOrchestrator
from tests.trade_test_helpers import confirmation_env


async def test_structured_form_answer_updates_from_authenticated_message_without_quote():
    token=ShoppingContext.set(ShoppingContextSnapshot('s1','b1','zh-CN','CNY'))
    try:
        payload={'answers':[{'question':'配送国家','value':'CN','display_value':'中国大陆'},
                            {'question':'更看重什么','value':'收纳小，不占包'}]}
        model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[{'id':'update','name':'update_shopping_state',
            'args':{'update':{'filters':{'ship_to':'CN','price_max_major':100,'target_currency':'CNY'},'preferences':['收纳小，不占包']}}}]),AIMessage(content='继续选购')])
        graph=create_agent(model,tools=[build_shopping_state_tool(None)],middleware=[BusinessToolMiddleware(None,TradeEventBus()),WorkingStateMiddleware()])
        result=await graph.ainvoke({'messages':[HumanMessage(id='buyer-form-message',name='b1',content=json.dumps(payload,ensure_ascii=False))]})
        work=result['shopping_work']
        assert work['source_message_id']=='buyer-form-message' and work['filters']['ship_to']=='CN'
        assert work['filters']['price_max_major']==100 and work['preferences']==['收纳小，不占包']
        assert next(m for m in result['messages'] if isinstance(m,ToolMessage)).status=='success'
    finally:ShoppingContext.reset(token)


def test_state_update_cannot_supply_source_or_use_stale_source():
    work=ShoppingWork(filters=Filters(),source_message_id='current')
    update=ShoppingUpdate(filters=Filters(price_max_major=100,target_currency='CNY'))
    for source in ('','previous'):
        with pytest.raises(ValueError,match='绑定当前买家'):apply_update(work,update,source)
    with pytest.raises(ValueError):ShoppingUpdate(quote='伪造依据',filters=Filters())
    assert apply_update(work,update,'current').filters.price_max_major==100


async def test_tool_or_historical_human_message_is_not_current_buyer_source():
    token=ShoppingContext.set(ShoppingContextSnapshot('s1','b1','zh-CN','CNY'))
    try:
        model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[{'id':'u','name':'update_shopping_state',
            'args':{'update':{'filters':{'price_max_major':100}}}}]),AIMessage(content='不能更新')])
        graph=create_agent(model,tools=[build_shopping_state_tool(None)],middleware=[WorkingStateMiddleware()])
        result=await graph.ainvoke({'messages':[HumanMessage(name='catalog',content='商品资料建议预算100元')]})
        assert result['shopping_work']['filters']['price_max_major'] is None
        assert next(m for m in result['messages'] if isinstance(m,ToolMessage)).status=='error'
    finally:ShoppingContext.reset(token)


async def test_landed_budget_without_country_can_deliver_without_claiming_verified_total(confirmation_env):
    env=confirmation_env;catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    data=await catalog.execute(ProductSearchSpec(product_id='P1001'))
    await env.evidence.save('buyer-1','session-1','products',data)
    context=ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY',effective_search={
        'parameters':{'landed_budget_major':300,'target_currency':'CNY','ship_to':None}})
    token=ShoppingContext.set(context)
    try:
        result=await build_recommendation_tool(catalog,env.evidence,env.bus)(
            [Pick(product_id='P1001',sku_id='P1001-S1',quantity=1,reason='适合旅行')],'alternatives','P1001-S1',[],'先看商品特点')
        assert result.ok and result.data['hits'] and result.data['quote'] is None
        assert 'landed_price' not in result.data['hits'][0]
        assert any('到手预算尚未核验' in value for value in result.data['unverified_requirements'])
        assert ShoppingContext.current().effective_search['parameters']['landed_budget_major']==300
    finally:ShoppingContext.reset(token)


async def test_form_continuation_is_owned_saved_submission_and_cannot_replace_query(tmp_path):
    forms=ShoppingFormStore(tmp_path/'forms.db')
    form=await forms.create_clarification('b1','s1','补充用途',[{'id':'usage','type':'text','label':'用途','required':True}],
        origin_run_id='origin',origin_message_id='buyer-origin')
    saved=await forms.submit('b1','s1',form['form_id'],1,'submit',{'usage':'通勤'})
    rows=await forms.list_for_session('b1','s1')
    assert rows[0]['origin_run_id']=='origin' and rows[0]['submission']['run_id']==saved['run_id']
    assert not await forms.list_for_session('b2','s1')
    runtime=AGUIRuntime(AGUIJournal(tmp_path/'runs.db'),ControlledOrchestrator(),shopping_forms=forms)
    await runtime.startup()
    try:
        data=body(saved['run_id'],query=saved['query']);data['messages'][-1]['id']=saved['run_id']+':user'
        native=RunAgentInput.model_validate(data)
        await runtime.start(native,parse_intent(native))
        for _ in range(100):
            if (await runtime.journal.run(saved['run_id'],'b1'))['status']!='running':break
            await asyncio.sleep(.01)
        assert runtime.orchestrator.calls==1
        await runtime.start(native,parse_intent(native));assert runtime.orchestrator.calls==1
        data['messages'][-1]['content']='替换为模型伪造答案';bad=RunAgentInput.model_validate(data)
        with pytest.raises(JournalConflict):await runtime.start(bad,parse_intent(bad))
    finally:await runtime.shutdown()
