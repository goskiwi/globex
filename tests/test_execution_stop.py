"""执行限制与部分交付：真实原生图、并行工具、checkpoint 和交易底座。"""
import asyncio
import json
from contextvars import ContextVar
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.agents.handoff import build_submission_tool
from app.application.runtime.errors import ExecutionStopped
from app.application.runtime.handoff import HandoffContext, HandoffMiddleware, TaskEvidence
from app.application.runtime.execution import ExecutionMiddleware
from app.application.runtime.middleware import BusinessToolMiddleware, ToolResilienceMiddleware
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.results import ToolResult
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.infrastructure.budget import init_budget
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.resilience import CircuitBreakerRegistry, DEFAULT_TIMEOUTS
from tests.test_agent_handoff import ScriptedModel, call, factories, task, submission
from tests.native_model_helpers import client_model
from tests.trade_test_helpers import confirmation_env, test_address as address


@pytest.fixture(autouse=True)
def scope():
    token=ShoppingContext.set(ShoppingContextSnapshot('handoff','buyer','zh-CN','CNY'))
    init_budget(0)
    yield
    init_budget(0)
    ShoppingContext.reset(token)


class StopModel(ScriptedModel):
    reason: str = 'budget_exhausted'
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        if self.cursor>=len(self.responses):
            raise ExecutionStopped(self.reason)
        return super()._generate(messages,stop,run_manager,**kwargs)


def root(graph,bus):
    orchestrator=MainAgentOrchestrator.__new__(MainAgentOrchestrator)
    orchestrator._bus=bus
    orchestrator._native_observer=ContextVar('stop-test-observer',default=None)
    session=SimpleNamespace(graph=graph,config={'configurable':{'thread_id':'handoff'}})
    return orchestrator,session


def product(bus,pid='P1001'):
    bus.publish('handoff','tool.result',{'tool':'product_search_tool','hits':[
        {'product_id':pid,'skus':[{'sku_id':pid+'-S1'}]}],'result_ref':'ctx_'+pid})


async def test_real_budget_denial_is_stop_not_submission_correction(tmp_path):
    calls=[]
    def forbidden(request):
        calls.append(1);raise AssertionError('不得请求模型')
    model=await client_model(tmp_path,forbidden)
    graph=create_agent(model,tools=[build_submission_tool()],middleware=[HandoffMiddleware()],
                       context_schema=HandoffContext)
    factory=SimpleNamespace(build=lambda:graph)
    dispatch=build_task_dispatch_tool(factory,factory,TradeEventBus())
    init_budget(1)
    try:
        result=(await dispatch('search_agent',task())).data
        assert result['status']=='failed' and result['stop_reason']=='budget_exhausted'
        assert result['feedback']==[] and result['issues']==['budget_exhausted']
        assert not calls and 'submission_missing' not in json.dumps(result)
    finally:await model.aclose()


async def test_child_keeps_evidence_not_unscreened_candidates_when_stopped(tmp_path,monkeypatch):
    factory,_,dispatch,bus=factories(tmp_path,monkeypatch,[])
    model=StopModel(responses=[call('read_candidates',{})])
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:model)
    result=(await dispatch('search_agent',task())).data
    assert result['status']=='partial' and result['stop_reason']=='budget_exhausted'
    assert result['candidates']==[] and result['observed_product_count']==2
    assert result['evidence_refs']==['ctx_demo'] and result['feedback']==[]
    assert model.cursor==1


async def test_child_step_limit_returns_partial_not_generic_execution_error(tmp_path,monkeypatch):
    factory,_,dispatch,bus=factories(tmp_path,monkeypatch,[])
    model=ScriptedModel(responses=[call('read_candidates',{}) for _ in range(20)])
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:model)
    build=factory.build
    factory.build=lambda:build().with_config({'recursion_limit':8})
    result=(await dispatch('search_agent',task())).data
    assert result['stop_reason']=='step_limit' and result['status']=='partial'
    assert result['candidates']==[] and result['observed_product_count']==2 and result['feedback']==[]


async def test_dispatch_owns_timeout_and_keeps_evidence_through_runtime(monkeypatch):
    bus=TradeEventBus()
    closed=asyncio.Event()
    class Child:
        async def ainvoke(self,*args,**kwargs):
            product(bus)
            try:await asyncio.Event().wait()
            finally:closed.set()
    factory=SimpleNamespace(build=lambda:Child())
    dispatch=build_task_dispatch_tool(factory,factory,bus)
    monkeypatch.setitem(DEFAULT_TIMEOUTS,'task_dispatch',.02)
    model=ScriptedModel(responses=[call('task_dispatch',{'subagent_type':'search_agent','task':task()}),
                                   AIMessage(content='部分结果')])
    graph=create_agent(model,tools=[as_langchain_tool(dispatch)],
                       middleware=[ToolResilienceMiddleware(CircuitBreakerRegistry()), ExecutionMiddleware(20,main=True)])
    result=await graph.ainvoke({'messages':[HumanMessage(content='研究')]})
    output=json.loads(next(m.content for m in result['messages'] if isinstance(m,ToolMessage)))
    assert output['stop_reason']=='timeout' and output['status']=='partial'
    assert output['candidates']==[] and output['observed_product_count']==1 and closed.is_set()


@pytest.mark.parametrize('second_status',['partial','failed'])
async def test_parallel_child_results_survive_parent_stop(second_status):
    bus=TradeEventBus()
    class Child:
        async def ainvoke(self,inputs,**kwargs):
            goal=inputs['handoff_task']['goal']
            if goal=='耳机' and second_status=='failed':
                raise RuntimeError('合成服务错误')
            product(bus,'P1001' if goal=='包' else 'P1002')
            await asyncio.sleep(0)
            if goal=='耳机':raise ExecutionStopped('step_limit')
            return {'handoff_result':{'status':'completed','summary':'完成',
                'candidates':[{'product_id':'P1001','reason':'已核验'}]}}
    factory=SimpleNamespace(build=lambda:Child())
    dispatch=build_task_dispatch_tool(factory,factory,bus)
    model=StopModel(responses=[AIMessage(content='',tool_calls=[
        {'id':'bag','name':'task_dispatch','args':{'subagent_type':'search_agent','task':task(goal='包')}},
        {'id':'ear','name':'task_dispatch','args':{'subagent_type':'search_agent','task':task(goal='耳机')}}])])
    graph=create_agent(model,tools=[as_langchain_tool(dispatch)],middleware=[BusinessToolMiddleware(None,bus)],checkpointer=InMemorySaver())
    runner,session=root(graph,bus)
    text=(await runner._reply('handoff',session,[HumanMessage(content='包和耳机')])).text
    assert 'P1001' in text and 'completed' in text and second_status in text
    assert ('P1002' in text)==(second_status=='partial')
    snapshot=await graph.aget_state(session.config)
    assert snapshot.next==() and snapshot.values['messages'][-1].additional_kwargs['execution_stop']=='budget_exhausted'
    outputs=[json.loads(m.content) for m in snapshot.values['messages'] if isinstance(m,ToolMessage)]
    assert {r['status'] for r in outputs}=={'completed',second_status}


async def test_main_no_results_stop_is_explicit_and_next_turn_works():
    model=StopModel()
    graph=create_agent(model,checkpointer=InMemorySaver())
    runner,session=root(graph,TradeEventBus())
    text=(await runner._reply('handoff',session,[HumanMessage(content='购物')])).text
    assert '尚无可核验' in text and '预算不足' in text
    assert (await graph.aget_state(session.config)).next==()
    model.responses=[AIMessage(content='新一轮')];model.cursor=0
    assert (await runner._reply('handoff',session,[HumanMessage(content='继续')])).text=='新一轮'


async def test_confirmed_preparation_survives_stop_without_replaying(confirmation_env):
    from app.application.tools.order_tools import build_create_order_tool
    from app.application.usecases.order_usecases import PlaceOrderUseCase
    env=confirmation_env
    ShoppingContext.set(replace(ShoppingContext.current(),selected_lines=(
        {'product_id':'P1001','sku_id':'P1001-S1','quantity':2},)))
    await env.evidence.save('buyer','handoff','products',{'hits':[{'product_id':'P1001','skus':[{'sku_id':'P1001-S1'}]}]})
    fn=build_create_order_tool(PlaceOrderUseCase(env.service),env.bus,env.evidence)
    model=StopModel(responses=[call('create_order_tool',{'sku_ids':['P1001-S1'],'shipping_address':asdict(address())})])
    graph=create_agent(model,tools=[as_langchain_tool(fn)],middleware=[BusinessToolMiddleware(None,env.bus)],checkpointer=InMemorySaver())
    runner,session=root(graph,env.bus)
    before=await env.store.get_inventory()
    text=(await runner._reply('handoff',session,[HumanMessage(content='准备确认卡')])).text
    assert '待用户确认，尚未执行交易' in text
    assert await env.store.get_inventory()==before
    confirmations=await env.store.list_confirmations(buyer_id='buyer',session_id='handoff')
    assert len(confirmations)==1
    model.responses=[AIMessage(content='等待用户确认')];model.cursor=0
    await runner._reply('handoff',session,[HumanMessage(content='先等等')])
    assert len(await env.store.list_confirmations(buyer_id='buyer',session_id='handoff'))==1


async def test_interrupted_tool_batch_is_closed_without_reexecution():
    bus=TradeEventBus();done=asyncio.Event();calls=[]
    async def read():
        """合成读取。"""
        calls.append('read');done.set();await asyncio.sleep(.02);product(bus);return ToolResult('已读取')
    async def stopped():
        """模拟工具内部依赖模型时预算耗尽。"""
        await done.wait();raise ExecutionStopped('budget_exhausted')
    model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[
        {'id':'a','name':'read','args':{}},{'id':'b','name':'stopped','args':{}}])])
    graph=create_agent(model,tools=[as_langchain_tool(read),as_langchain_tool(stopped)],
        middleware=[ToolResilienceMiddleware(CircuitBreakerRegistry()), ExecutionMiddleware(20)],checkpointer=InMemorySaver())
    runner,session=root(graph,bus)
    text=(await runner._reply('handoff',session,[HumanMessage(content='读取')])).text
    snapshot=await graph.aget_state(session.config)
    messages=snapshot.values['messages']
    assert 'P1001' in text and snapshot.next==() and calls==['read']
    tool_ids={c['id'] for m in messages if isinstance(m,AIMessage) for c in m.tool_calls}
    assert tool_ids=={m.tool_call_id for m in messages if isinstance(m,ToolMessage)}
    model.responses=[AIMessage(content='继续新任务')];model.cursor=0
    await runner._reply('handoff',session,[HumanMessage(content='继续')])
    assert calls==['read']


async def test_cancel_is_not_relabelled_as_timeout():
    class Child:
        async def ainvoke(self,*args,**kwargs):raise asyncio.CancelledError()
    factory=SimpleNamespace(build=lambda:Child())
    dispatch=build_task_dispatch_tool(factory,factory,TradeEventBus())
    with pytest.raises(asyncio.CancelledError):
        await dispatch('search_agent',task())


async def test_main_step_limit_keeps_results_and_closes_checkpoint():
    bus=TradeEventBus()
    async def read():
        """反复读取相同测试数据。"""
        product(bus);return ToolResult('已读取')
    model=ScriptedModel(responses=[call('read',{}) for _ in range(100)])
    graph=create_agent(model,tools=[as_langchain_tool(read)],checkpointer=InMemorySaver())
    runner,session=root(graph,bus)
    session.config['recursion_limit']=40
    text=(await runner._reply('handoff',session,[HumanMessage(content='研究')])).text
    assert '步数上限' in text and 'P1001' in text
    snapshot=await graph.aget_state(session.config)
    assert snapshot.next==() and snapshot.values['messages'][-1].additional_kwargs['execution_stop']=='step_limit'


async def test_budget_stop_in_mixed_submit_batch_cannot_be_reported_completed(tmp_path,monkeypatch):
    from langchain_core.tools import tool
    factory,_,dispatch,bus=factories(tmp_path,monkeypatch,[])
    @tool
    async def stop_tool():
        """模拟业务工具依赖的模型预算耗尽。"""
        raise ExecutionStopped('budget_exhausted')
    original=factory.build_tools
    factory.build_tools=lambda:[*original(),stop_tool]
    model=ScriptedModel(responses=[call('read_candidates',{}),AIMessage(content='',tool_calls=[
        call('stop_tool',{}).tool_calls[0],
        submission(candidates=[{'product_id':'P1001','reason':'候选'}]).tool_calls[0]])])
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:model)
    result=(await dispatch('search_agent',task())).data
    assert result['status']=='partial' and result['stop_reason']=='budget_exhausted'
    assert result['feedback']==[] and model.cursor==2


async def test_stream_denial_produces_no_fabricated_assistant_text(tmp_path):
    def forbidden(request):raise AssertionError('不得请求上游')
    model=await client_model(tmp_path,forbidden)
    init_budget(1);chunks=[]
    try:
        with pytest.raises(ExecutionStopped):
            async for chunk in model.astream('查询'):chunks.append(chunk)
        assert chunks==[]
    finally:await model.aclose()


async def test_stopped_checkpoint_reopens_from_sqlite_as_closed_turn(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    path=str(tmp_path/'graph.db')
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph=create_agent(StopModel(),checkpointer=saver)
        runner,session=root(graph,TradeEventBus())
        await runner._reply('handoff',session,[HumanMessage(content='原始任务')])
        assert (await graph.aget_state(session.config)).next==()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph=create_agent(ScriptedModel(responses=[AIMessage(content='新任务完成')]),checkpointer=saver)
        runner,session=root(graph,TradeEventBus())
        before=await graph.aget_state(session.config)
        assert before.next==() and before.values['messages'][-1].additional_kwargs['execution_stop']=='budget_exhausted'
        assert (await runner._reply('handoff',session,[HumanMessage(content='新的请求')])).text=='新任务完成'
