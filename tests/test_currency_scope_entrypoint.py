"""从真实HTTP入口、模型SDK与状态编译验证默认显示币种和明确约束。"""
import json
from dataclasses import replace

import httpx
import pytest

from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
from app.application.runtime.research_plan import condition_identity
from scripts.eval.agent_runtime import isolated_runtime
from scripts.eval.trace import tool_observations, presentations
from scripts.run_stream import run_input, read_stream
from tests.native_model_helpers import client_model, completion
from tests.test_retrieval import _settings


def test_compiled_currency_keeps_default_separate_from_explicit_constraint():
    empty = compile_search(ShoppingWork(filters=Filters()), currency='CNY')
    explicit = compile_search(ShoppingWork(filters=Filters(target_currency='CNY')), currency='CNY')
    assert empty['parameters']==explicit['parameters']
    assert empty['currency_constraint'] is None and explicit['currency_constraint']=='CNY'
    assert condition_identity(empty)!=condition_identity(explicit)


@pytest.mark.parametrize('display,filters,quote_currency,budget,expected_amount',[
    ('CNY',{},'USD',100,2169),
    ('USD',{},'CNY',300,15400),
    ('CNY',{'target_currency':'USD'},'USD',100,2169),
    ('CNY',{'target_currency':'CNY'},'USD',100,None),
    ('CNY',{'target_currency':'USD','landed_budget_major':20},'USD',100,None),
    ('CNY',{'target_currency':'USD','ship_to':'JP'},'USD',100,None),
])
async def test_bundle_currency_through_http_model_and_real_compiler(tmp_path,display,filters,quote_currency,budget,expected_amount):
    requests=[]
    search={'sku_id':'P1003-S1'}
    if filters:search['conditions']={'filters':filters}
    bundle={'picks':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'合成已核验候选'}],
            'guidance':'合成组合报价','ship_to':'CN','currency':quote_currency,'landed_budget_major':budget,'turn_action':'finish'}
    def handler(request):
        requests.append(json.loads(request.content));raw=completion()
        index=len(requests)-1
        if index<2:
            name,args=[('product_search_tool',search),('recommend_bundle',bundle)][index]
            raw['choices'][0].update(finish_reason='tool_calls',message={'role':'assistant','content':None,
                'tool_calls':[{'id':f'call-{index}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]})
        else:
            assert index==2
            raw['choices'][0]['message']['content']='当前明确约束不允许该报价，未交付或交易。'
        return httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler)
    try:
        async with isolated_runtime(tmp_path/'runtime',_settings(tmp_path),model_factory=lambda *a,**k:model) as (app,container):
            headers={'Authorization':'Bearer '+app.state.identity_policy.issue('synthetic-buyer')}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers=headers) as client:
                body=run_input('currency-session','synthetic-buyer','按给定约束核验一件P1003-S1并交付组合报价，不下单')
                body['forwardedProps']['currency']=display
                response=await client.post('/commerce/ag-ui/run',json=body)
                assert response.status_code==200,response.text
                stream=read_stream(response.text);stream.require_success()
                observations=tool_observations(stream.frames)
                assert [o['name'] for o in observations]==['product_search_tool','recommend_bundle']
                result=observations[-1]
                deliveries=presentations(stream.frames,'recommendation')
                if expected_amount is None:
                    assert result['status']=='error' and not deliveries
                    assert '不一致' in result['data'] or '超过当前预算' in result['data']
                else:
                    assert result['status']=='success',result
                    quote=deliveries[-1][1]['quote']
                    assert quote['currency']==quote_currency and quote['total_amount_minor']==expected_amount
                    assert len(requests)==2
                session=container.orchestrator._sessions._agents['currency-session']
                state=(await session.graph.aget_state(session.config)).values
                assert state['shopping_work']['filters']['target_currency']==filters.get('target_currency')
                assert not await container.trade_store.list_confirmations(buyer_id='synthetic-buyer',session_id='currency-session')
                assert (await container.trade_store.list_orders(buyer_id='synthetic-buyer'))['total']==0
    finally:
        await model.aclose()


def test_clear_currency_restores_default_without_rewriting_requested_filters():
    from app.application.agents.shopping_state import ShoppingUpdate,apply_update
    original=ShoppingWork(filters=Filters(target_currency='USD'),source_message_id='buyer-message')
    cleared=apply_update(original,ShoppingUpdate(filters=Filters(target_currency=None)),'buyer-message')
    policy=compile_search(cleared,currency='CNY')
    assert policy['currency_constraint'] is None and policy['parameters']['target_currency']=='CNY'
    assert original.filters.target_currency=='USD'


@pytest.mark.parametrize('value',['','  ','usd','XXX'])
def test_explicit_currency_requires_domain_code_or_null(value):
    from pydantic import ValidationError
    from app.application.tools.recommendation_tools import QuoteTerms
    with pytest.raises(ValidationError):Filters(target_currency=value)
    with pytest.raises(ValidationError):QuoteTerms(ship_to='CN',currency=value)


async def test_currency_clear_across_http_turns_preserves_unknown_budget_units(tmp_path):
    calls=[]
    picks=[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'合成已核验候选'}]
    responses=[
        ('product_search_tool',{'sku_id':'P1003-S1','conditions':{'filters':{'target_currency':'USD','landed_budget_major':100}}}),
        ('recommend_bundle',{'picks':picks,'guidance':'明确美元预算内的报价','ship_to':'CN','currency':'USD','landed_budget_major':100,'turn_action':'finish'}),
        ('product_search_tool',{'sku_id':'P1003-S1','conditions':{'filters':{'target_currency':None}}}),
        ('recommend_bundle',{'picks':picks,'guidance':'未重新解释旧预算币种，只展示这次人民币报价','ship_to':'CN','currency':'CNY','landed_budget_major':None,'turn_action':'finish'}),
    ]
    def handler(request):
        calls.append(json.loads(request.content));name,args=responses[len(calls)-1];raw=completion()
        raw['choices'][0].update(finish_reason='tool_calls',message={'role':'assistant','content':None,'tool_calls':[
            {'id':f'call-{len(calls)}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]})
        return httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler)
    try:
        async with isolated_runtime(tmp_path/'runtime',_settings(tmp_path),model_factory=lambda *a,**k:model) as (app,container):
            headers={'Authorization':'Bearer '+app.state.identity_policy.issue('synthetic-buyer')}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers=headers) as client:
                for index,query in enumerate(['100美元到手预算，核验P1003-S1后报价，不下单','清除之前币种限制，不修改预算数字；请用人民币展示当前报价，不下单']):
                    response=await client.post('/commerce/ag-ui/run',json=run_input('clear-session','synthetic-buyer',query))
                    assert response.status_code==200,response.text
                    stream=read_stream(response.text);stream.require_success()
                    assert all(r['status']=='success' for r in tool_observations(stream.frames))
                    panel=presentations(stream.frames,'recommendation')[-1][1]
                    assert panel['quote']['currency']==('USD' if index==0 else 'CNY')
                session=container.orchestrator._sessions._agents['clear-session']
                state=(await session.graph.aget_state(session.config)).values
                work=ShoppingWork.model_validate(state['shopping_work'])
                assert work.filters.target_currency is None and work.filters.landed_budget_major==100
                policy=compile_search(work,currency='CNY')
                assert policy['currency_constraint'] is None and policy['parameters']['landed_budget_major'] is None
                assert '到手预算币种未明确，尚未执行总价上限' in panel['unverified_requirements']
                assert panel['quote']['total_amount_minor']==15400 and len(calls)==4
                assert (await container.trade_store.list_orders(buyer_id='synthetic-buyer'))['total']==0
    finally:await model.aclose()
