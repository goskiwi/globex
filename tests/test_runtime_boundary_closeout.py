"""八项边界缺陷：业务、原生图、AG-UI 和应用生命周期组合验证。"""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from ag_ui.core import RunAgentInput
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from openai import AsyncOpenAI

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.runtime.context import RequestComposition
from app.application.runtime.execution import ExecutionMiddleware
from app.application.runtime.handoff import TaskEvidence
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.results import ToolResult, ToolResultState
from app.application.tools.order_tools import build_query_order_tool
from app.application.usecases.order_usecases import QueryOrderUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.operational_metrics import begin_request, finish_request
from app.presentation.ag_ui import parse_intent
from scripts.eval.interview_runtime import isolated_runtime
from tests.native_model_helpers import client_model, completion
from tests.native_tool_helpers import tool_graph, call_tool
from tests.test_ag_ui_journal import body
from tests.test_execution_stop import root
from tests.test_agent_handoff import ScriptedModel
from tests.test_retrieval import _settings


@pytest.fixture
def scope():
    token=ShoppingContext.set(ShoppingContextSnapshot('s1','b1','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)


@pytest.mark.parametrize('stream',[False,True])
async def test_truncated_text_is_partial_not_completed(tmp_path,stream):
    def handler(request):
        raw=completion();raw['choices'][0]['message']['content']='第一款是'
        raw['choices'][0]['finish_reason']='length'
        if not stream:return httpx.Response(200,json=raw)
        chunk={**raw,'object':'chat.completion.chunk','choices':[{
            'index':0,'delta':{'content':'第一款是'},'finish_reason':'length'}]}
        return httpx.Response(200,headers={'content-type':'text/event-stream'},
            content='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
    model=await client_model(tmp_path,handler,stream=stream)
    graph=create_agent(model,middleware=[ExecutionMiddleware(4,main=True)],checkpointer=InMemorySaver())
    runner,session=root(graph,TradeEventBus())
    try:
        result=await runner._reply('s1',session,[HumanMessage(content='推荐')])
        assert result.status=='partial' and result.stop_reason=='output_limit'
        assert '第一款是' in result.text
    finally:await model.aclose()


@pytest.mark.parametrize('compact',[False,True])
async def test_inactive_skill_stays_out_of_final_request(compact):
    data={'kind':'skill','body':'OLD_SKILL_BODY','id':'demo'}
    message=ToolMessage(name='load_agent_skill_tool',tool_call_id='old',
        content=json.dumps(data),artifact={'data':data})
    original=message.model_dump_json()
    request=ModelRequest(model=ScriptedModel(responses=[]),messages=[message],tools=[],state={})
    settings=SimpleNamespace(context_size=128000,context_compact_result_rules=compact)
    final=await RequestComposition(request,settings,[],None).render([message])
    assert all('OLD_SKILL_BODY' not in m.text for m in final.request.messages)
    assert message.model_dump_json()==original


async def test_skill_success_with_notice_uses_structured_event(scope):
    from app.application.harness.loop_detector import LoopDetector
    bus=TradeEventBus();adapter=AGUIRunAdapter(RunAgentInput.model_validate(body()),lambda e:None)
    async def load_agent_skill_tool(skill_id:str,version:str):
        """读取合成 Skill。"""
        return ToolResult({'id':skill_id,'version':version,'kind':'skill','title':'演示',
            'body':'参考流程','content_hash':'a'*64,'authority':'reference_only'})
    graph=tool_graph(load_agent_skill_tool,middlewares=[BusinessToolMiddleware(LoopDetector(repeat_threshold=2),bus)])
    with observe_run_events(adapter.on_trade_event):
        await call_tool(graph,skill_id='demo',version='1')
        receipt=await call_tool(graph,skill_id='demo',version='1')
    assert receipt.artifact['notices']
    assert all(s['status']=='used' for s in adapter.state['skillUsages'])


async def test_order_query_survives_in_handoff_and_partial_delivery(scope):
    bus=TradeEventBus();evidence=TaskEvidence('s1')
    store=SimpleNamespace(get_order=AsyncMock(return_value={'order_id':'GBX-AUDIT','status':'CONFIRMED','items':[]}))
    graph=tool_graph(build_query_order_tool(QueryOrderUseCase(store),bus),middlewares=[BusinessToolMiddleware(None,bus)])
    with observe_run_events(evidence.capture):await call_tool(graph,order_id='GBX-AUDIT')
    assert evidence.trade==[{'order_id':'GBX-AUDIT','status':'CONFIRMED'}]
    assert 'GBX-AUDIT' in evidence.stopped_text('model_call_limit')


async def test_parallel_graphs_with_same_provider_call_id_do_not_collide(scope):
    bus=TradeEventBus();events=[];adapter=AGUIRunAdapter(RunAgentInput.model_validate(body()),events.append)
    async def read_bag():
        """读取包。"""
        await asyncio.sleep(.01);return '包资料'
    async def read_audio():
        """读取耳机。"""
        await asyncio.sleep(.02);return '耳机资料'
    graphs=[tool_graph(fn,middlewares=[BusinessToolMiddleware(None,bus)]) for fn in (read_bag,read_audio)]
    with observe_run_events(adapter.on_trade_event):await asyncio.gather(*(call_tool(g) for g in graphs))
    results=[e for e in events if e.type=='TOOL_CALL_RESULT']
    assert len({e.tool_call_id for e in results})==2
    assert len(adapter.state['process']['steps'])==2
    assert {e.content for e in results}=={'包资料','耳机资料'}


async def test_tool_metrics_count_registered_calls_once_including_failures(scope):
    async def quote_products():
        """返回业务拒绝。"""
        return ToolResult('目的地不支持',state=ToolResultState.ERROR,error_code='business_rejected')
    bus=TradeEventBus();observation=begin_request()
    try:
        await call_tool(tool_graph(quote_products,middlewares=[BusinessToolMiddleware(None,bus)]))
    finally:result=finish_request(observation,'failed')
    assert result['tool_calls']==1 and result['tool_errors']==1


@pytest.mark.parametrize('outcome',['normal','error','cancel'])
async def test_production_owns_clients_and_never_publishes_raw_text(tmp_path,monkeypatch,outcome):
    import app.infrastructure.llm as llm
    clients=[];waiting=asyncio.Event();synthetic='sk-'+'a'*24
    class Wire(httpx.AsyncByteStream):
        async def __aiter__(self):
            for part in (synthetic[:6],synthetic[6:]):
                yield ('data: '+json.dumps({'id':'a','object':'chat.completion.chunk','created':1,'model':'test',
                    'choices':[{'index':0,'delta':{'content':part},'finish_reason':None}]})+'\n\n').encode()
            if outcome=='error':raise httpx.ReadError('合成连接中断')
            if outcome=='cancel':
                waiting.set();await asyncio.Event().wait()
            yield ('data: '+json.dumps({'id':'a','object':'chat.completion.chunk','created':1,'model':'test',
                'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})+'\n\ndata: [DONE]\n\n').encode()
    def sdk(**kwargs):
        client=AsyncOpenAI(**{**kwargs,'http_client':httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req:httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Wire())))})
        clients.append(client);return client
    monkeypatch.setattr(llm,'AsyncOpenAI',sdk)
    settings=replace(_settings(tmp_path),llm_api_key='test-key',embedding_api_key='test-key',llm_base_url='https://audit.test/v1',output_guard_enabled=True)
    try:
        async with isolated_runtime(tmp_path/'runtime',settings) as (_,container):
            request=RunAgentInput.model_validate(body());runtime=container.ag_ui_runtime
            await runtime.start(request,parse_intent(request))
            if outcome=='cancel':
                await asyncio.wait_for(waiting.wait(),5)
                await runtime.cancel('r1','b1')
            async with asyncio.timeout(5):
                while (await runtime.journal.run('r1','b1'))['status']=='running':await asyncio.sleep(.01)
            saved=await runtime.journal.run('r1','b1')
            assert saved['status']=={'normal':'completed','error':'error','cancel':'stopped'}[outcome]
            assert len(clients)==1, '主图、摘要和偏好提炼应复用入口客户端'
            events,_,_=await runtime.journal.events('r1','b1',0)
            assert synthetic not in json.dumps(events,ensure_ascii=False)
            assert synthetic not in ''.join(e['event'].get('delta','') for e in events)
            await container.shutdown()
            assert clients and all(c.is_closed() for c in clients)
    finally:
        for c in clients:await c.close()
