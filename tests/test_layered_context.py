"""原生上下文策略、SQLite 整理操作及原文作用域验证。"""
import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import select
from app.infrastructure.persistence.sql.session_store import SqlFencedSessionStore,ContextCheckpointRow,ContextOperationRow
from app.domain.session.ports.session_store import StaleSessionWrite,SessionOwnerMismatch
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from app.infrastructure.context import ShoppingContext,ShoppingContextSnapshot
from app.infrastructure.context_products import business_view,product_page,token_estimate,result_identity
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from app.application.runtime.working_state import WorkingStateMiddleware
from tests.shopping_state_helpers import work_fixture
from app.application.runtime.projections import read_output
from tests.native_context_helpers import history,policy,apply

@pytest.fixture(autouse=True)
def scope():
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)

def test_entry_keeps_business_description_skus_and_tariff():
    raw={'hits':[{'product_id':'P1','description':'不可用于食品','image_url':'https://image',
                 'image_kind':'illustration','image_alt':'商品图片','skus':[{'sku_id':'S1'},{'sku_id':'S2'}],'landed_price':{'tax':3}}]}
    view=business_view(raw)
    assert view['hits'][0]['description']=='不可用于食品'
    assert len(view['hits'][0]['skus'])==2 and view['hits'][0]['landed_price']['tax']==3
    for field in ('image_url', 'image_kind', 'image_alt'):
        assert field not in view['hits'][0] and field in raw['hits'][0]
        assert field not in product_page(raw, fields='all')['hits'][0]

def test_pagination_preserves_order_and_large_field_can_be_reassembled():
    data={'hits':[{'product_id':f'P{i}','description':'材料'*15000} for i in range(7)]}
    fragments=[];offset=0
    while True:
        page=product_page(data,limit=1,token_limit=3000,field_offset=offset)
        assert token_estimate(page)<3000
        fragments.append(page['fragment'])
        if page['next_field_offset'] is None:break
        offset=page['next_field_offset']
    assert json.loads(''.join(fragments))==data['hits'][0]
    assert page['next_offset']==1

async def test_batch_lookup_reopens_and_scopes_by_buyer(tmp_path):
    store=ContextEvidenceStore(tmp_path/'e.db')
    for ids in [['P1','P2'],['P3','P4']]:
        await store.save('b','s','display_batch',{'hits':[{'product_id':x} for x in ids]})
    fn=build_conversation_fact_lookup(ContextEvidenceStore(store.path))
    result=await fn(batch=1,position=2)
    assert json.loads(result.text)['records'][0]['data']['hits']==[{'product_id':'P2'}]
    other=ShoppingContext.set(ShoppingContextSnapshot('s','other','zh-CN','CNY'))
    try:
        result=await fn(batch=1)
        assert json.loads(result.text)['records']==[]
    finally:ShoppingContext.reset(other)

@pytest.mark.parametrize('change',[{'ship_to':'JP'},{'currency':'USD'},{'quantity':2}])
def test_quote_conditions_cannot_be_deduplicated_across_scopes(change):
    from copy import deepcopy
    first={'product_id':'P1','skus':[{'sku_id':'P1-S1'}],'landed_price':{'ship_to':'CN','currency':'CNY','quantity':1}}
    second=deepcopy(first);second['landed_price'].update(change)
    assert result_identity(first,{})!=result_identity(second,{})

def test_query_currency_and_direct_sku_are_part_of_quote_key():
    a={'product_id':'P1','sku_id':'P1-S1'}
    assert result_identity(a,{'currency':'CNY'})!=result_identity(a,{'currency':'USD'})
    assert result_identity(a,{})!=result_identity({**a,'sku_id':'P1-S2'}, {})

async def test_unread_recent_turns_selection_and_no_pressure(tmp_path):
    state=history();mw=policy(tmp_path,product_tokens=1)
    assert not (await mw.compact_checkpoint({**state,'read_tool_messages':[]}))['messages']
    updates=await mw.compact_checkpoint(state)
    assert len(updates['messages'])==6
    assert not (await policy(tmp_path,product_tokens=999999).compact_checkpoint(state))['messages']
    for message in state['messages']:
        if isinstance(message,ToolMessage):
            message.content=message.content.replace('P0','P1000')
            message.artifact={'data':json.loads(message.content)}
    selected=await mw.compact_checkpoint({**state,'shopping_work':work_fixture(selected=['P1000-S1'])})
    assert 't0' not in [m.id for m in selected['messages']]
    assert all(m.id not in ['t6','t7'] for m in updates['messages'])

async def test_evidence_failure_does_not_mutate_messages(tmp_path):
    state=history();before=[m.model_dump() for m in state['messages']]
    mw=policy(tmp_path,product_tokens=1)
    mw.store.save=AsyncMock(side_effect=OSError('disk'))
    with pytest.raises(OSError):await mw.compact_checkpoint(state)
    assert [m.model_dump() for m in state['messages']]==before

async def test_summary_invalid_sources_roll_back_and_stop_automatic_retries(tmp_path):
    state=history();mw=policy(tmp_path,target_tokens=10,response='选中P999999-S1')
    original=[m.model_dump() for m in state['messages']]
    for _ in range(3):
        update=await mw.compact_checkpoint(state)
        assert update['context_statistics']['status']=='failed'
        state=apply(state,update)
    assert [m.model_dump() for m in state['messages']]==original
    assert mw.model.ainvoke.await_count==6
    await mw.compact_checkpoint(state)
    assert mw.model.ainvoke.await_count==6

# 新需求操作与语言理解在 test_working_state_questions / 真实语言验证中覆盖。

async def test_multiple_searches_in_recent_turns_and_selected_duplicates(tmp_path):
    state=history()
    state['messages']=[m for m in state['messages'] if m.id not in {'u6','u7'}]
    updates=await policy(tmp_path,product_tokens=1).compact_checkpoint(state)
    assert [m.id for m in updates['messages']]==['t0','t1','t2','t3','t4','t5']
    state=history()
    for m in state['messages']:
        if isinstance(m,ToolMessage):
            m.content=json.dumps({'hits':[{'product_id':'P1001','skus':[{'sku_id':'P1001-S1'}]}],'query_conditions':{'ship_to':'JP'}})
            m.artifact={'data':json.loads(m.content)}
    state['shopping_work']=work_fixture(selected=['P1001-S1'])
    updates=await policy(tmp_path,product_tokens=1).compact_checkpoint(state)
    assert len(updates['messages'])==6 and 't7' not in [m.id for m in updates['messages']]

@pytest.mark.parametrize('window',[128000,4096,64000])
async def test_manual_api_operation_and_owner_are_persistent(tmp_path,monkeypatch,window):
    import httpx
    from fastapi import FastAPI
    from tests.test_langgraph_runtime import _container,ScriptedModel
    from app.application.agents.orchestrator import SubmitIntentInput
    from app.presentation.context_workspace import register_context_routes
    from app.infrastructure.prompt_registry import PromptContractChanged
    container=await _container(tmp_path,monkeypatch,ScriptedModel())
    try:
        assert not (await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','你好'))).error
        service=container.context_service
        container.orchestrator._sessions._main_factory._settings=replace(container.settings,context_size=window)
        if window==64000:
            async def denied(*args):raise PromptContractChanged('old contract')
            monkeypatch.setattr(container.orchestrator._sessions,'get_or_create',denied)
        api=FastAPI();api.state.session_store=service.store
        register_context_routes(api,lambda:service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api),base_url='http://test') as client:
            original=(await client.get('/commerce/context?buyer_id=b&session_id=s')).json()
            payload={'buyer_id':'b','session_id':'s','request_id':'op','expected_revision':original['revision']}
            response=await client.post('/commerce/context/compact',json=payload)
            assert response.status_code==202,response.text
            op=response.json()['operation_id']
            await asyncio.gather(*list(service.tasks.values()))
            result=(await client.get(f'/commerce/context/operations/{op}?buyer_id=b')).json()
            assert result['status']==('noop' if window==128000 else 'failed'),result
            if window!=128000:assert result['error_code']==('ContextCapacityError' if window==4096 else 'PromptContractChanged')
            assert (await client.get(f'/commerce/context/operations/{op}?buyer_id=other')).status_code==403
            assert (await client.post('/commerce/context/compact',json=payload)).json()['operation_id']==op
    finally:await container.shutdown()

async def test_native_lookup_uses_only_explicit_arguments_and_owner(tmp_path):
    from langchain.agents.middleware import ToolCallRequest
    from app.application.runtime.tools import as_langchain_tool
    store=ContextEvidenceStore(tmp_path/'scope.db')
    refs=[await store.save('b','s','display_batch',{'hits':[{'product_id':'P1','price_major':price}]}) for price in (10,20)]
    lookup=build_conversation_fact_lookup(store)
    request=ToolCallRequest(tool_call={'id':'lookup','name':'conversation_fact_lookup','args':{}},
        tool=as_langchain_tool(lookup),state={'shopping_work':{**work_fixture(), 'latest_request':'第一批的报价'},'messages':[HumanMessage(name='b',content=[
            {'type':'text','text':'第一批的报价'},{'type':'text','text':'附件中第二批不参与解析'}])]},runtime=None)
    async def handler(request):
        assert json.loads((await lookup(product_id='P1')).text)['records'][0]['data']['hits'][0]['price_major']==20
        assert json.loads((await lookup(batch=2)).text)['records'][0]['data']['hits'][0]['price_major']==20
        token=ShoppingContext.set(ShoppingContextSnapshot('other','b','zh-CN','CNY'))
        try:assert json.loads((await lookup(product_id='P1')).text)['records']==[]
        finally:ShoppingContext.reset(token)
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):await WorkingStateMiddleware().awrap_tool_call(request,handler)


async def test_read_markers_only_follow_successful_native_model_call(tmp_path):
    import httpx
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from tests.native_model_helpers import client_model,completion
    from openai import APIStatusError
    failing=True
    def handler(request):
        return httpx.Response(503,json={'error':{'message':'service unavailable'}}) if failing else httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    mw=policy(tmp_path)
    graph=create_agent(model,middleware=[mw],checkpointer=InMemorySaver())
    config={'configurable':{'thread_id':'s'}}
    try:
        with pytest.raises(APIStatusError):await graph.ainvoke({**history(1),'read_tool_messages':[]},config)
        assert not (await graph.aget_state(config)).values.get('read_tool_messages')
        failing=False
        await graph.ainvoke(None,config)
        assert (await graph.aget_state(config)).values['read_tool_messages']==['t0']
    finally:await model.aclose()


async def test_after_use_and_pressure_are_distinct_but_both_protect_unread(tmp_path):
    state=history()
    pressure=policy(tmp_path,product_tokens=999999)
    eager=policy(tmp_path,product_tokens=999999,context_pruning_timing='after_use')
    assert not (await pressure.compact_checkpoint(state))['messages']
    assert len((await eager.compact_checkpoint(state))['messages'])==6
    assert not (await eager.compact_checkpoint({**state,'read_tool_messages':[]}))['messages']


async def test_complete_request_capacity_includes_tools_before_http(tmp_path):
    from langchain.agents.middleware import ModelRequest
    from tests.test_langgraph_runtime import ScriptedModel
    from app.application.runtime.errors import ContextCapacityError
    mw=policy(tmp_path)
    mw.settings=replace(mw.settings,context_size=14000)
    request=ModelRequest(model=ScriptedModel(),messages=[HumanMessage(content='查询')],
        tools=[{'type':'function','function':{'name':'tool','description':'冗长声明'*10000,'parameters':{'type':'object','properties':{}}}}],state={},runtime=None)
    handler=AsyncMock()
    with pytest.raises(ContextCapacityError):await mw.awrap_model_call(request,handler)
    handler.assert_not_awaited()


async def test_checkpoint_and_state_atomic_fenced_and_operation_idempotent(tmp_path):
    engine=create_async_engine('sqlite+aiosqlite:///'+str(tmp_path/'db'))
    store=SqlFencedSessionStore(engine)
    try:
        claim=await store.claim('s',buyer_id='b')
        state_json=json.dumps({'middle_context':{'globex_context':{'checkpoint_id':'cp1','working':{'goal':'背包'}}}})
        newer=await store.claim('s',buyer_id='b')
        with pytest.raises(StaleSessionWrite):await store.save_claim(claim,state_json)
        async with engine.connect() as db:assert (await db.execute(select(ContextCheckpointRow))).first() is None
        saved=await store.save_claim(newer,state_json)
        assert (await store.context_view('s','b'))['checkpoint_id']=='cp1'
        op,created=await store.create_context_operation('s','b','request',saved.revision)
        assert created
        assert (await store.create_context_operation('s','b','request',saved.revision))[1] is False
        with pytest.raises(StaleSessionWrite):await store.create_context_operation('s','b','request',saved.revision+1)
        with pytest.raises(SessionOwnerMismatch):await store.context_operation(op['operation_id'],'other')
        from sqlalchemy import update
        async with engine.begin() as db:await db.execute(update(ContextOperationRow).values(deadline=0))
        await store.recover_context_operations()
        assert (await store.context_operation(op['operation_id'],'b'))['status']=='interrupted'
    finally:await engine.dispose()
