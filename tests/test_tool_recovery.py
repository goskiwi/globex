"""模块三：原生工具错误元数据、单层模型重试与业务进展。"""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
import pytest
from langchain_core.messages import ToolMessage
from app.application.runtime.results import ToolResult, ToolResultState
from app.application.runtime.middleware import ToolResilienceMiddleware, GatewayModelMiddleware
from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.harness.loop_detector import LoopDetector
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.transient import is_transient_error
from tests.native_tool_helpers import call_tool, tool_graph


@pytest.mark.parametrize("message", ["超时", "503 Service Unavailable", "timeout", "库存不足"])
async def test_business_rejection_is_not_classified_by_language(message):
    async def rejected():
        """明确业务拒绝，不是网络故障。"""
        return ToolResult(message, state=ToolResultState.ERROR, error_code="business_rejected")
    registry=CircuitBreakerRegistry(failure_threshold=1)
    graph=tool_graph(rejected,middlewares=[ToolResilienceMiddleware(registry)])
    for _ in range(3):
        result=await call_tool(graph)
        assert result.status=='error' and result.artifact['error_code']=='business_rejected'
    assert registry.status('rejected')=='closed'


async def test_rejection_does_not_clear_previous_service_failures():
    registry=CircuitBreakerRegistry()
    registry.record_failure('rejected')
    async def rejected():
        """业务拒绝。"""
        return ToolResult('需要地址', state=ToolResultState.ERROR, error_code='business_rejected')
    await call_tool(tool_graph(rejected,middlewares=[ToolResilienceMiddleware(registry)]))
    assert registry._state('rejected').consecutive_failures==1


async def test_error_prefix_in_success_is_not_execution_failure():
    async def literal():
        """正常引用文字。"""
        return ToolResult('[error] 是本文解释的字面标记')
    result=await call_tool(tool_graph(literal,middlewares=[ToolResilienceMiddleware(CircuitBreakerRegistry())]))
    assert result.status=='success'


@pytest.mark.parametrize("name",["create_order_tool","cancel_order_tool","remember_preference_tool","task_dispatch"])
async def test_runtime_never_replays_side_effect_or_dispatch(name):
    calls=[]
    async def uncertain():
        """可能已经执行，响应中断。"""
        calls.append(1)
        raise httpx.ReadTimeout('synthetic timeout')
    uncertain.__name__=name
    result=await call_tool(tool_graph(uncertain,middlewares=[ToolResilienceMiddleware(CircuitBreakerRegistry())]))
    assert calls==[1] and result.status=='error'
    assert result.artifact['error_code']=='unavailable'


async def test_only_model_layer_retries_and_outer_turn_does_not(monkeypatch):
    monkeypatch.setattr(asyncio,'sleep',AsyncMock())
    attempts=[]
    async def handler(request):
        attempts.append(1)
        raise httpx.ReadTimeout('synthetic timeout')
    middleware=GatewayModelMiddleware(SimpleNamespace(llm_fallback_model='',llm_model='test',llm_max_retries=2),None, client=None)
    async def invoke(*args):return await middleware.awrap_model_call(None,handler)
    orchestrator=SimpleNamespace(_invoke_graph=invoke)
    with pytest.raises(httpx.ReadTimeout):
        await MainAgentOrchestrator._reply(orchestrator,'s',None,[])
    assert len(attempts)==3
    assert not hasattr(MainAgentOrchestrator,'_reply_with_retry')


@pytest.mark.parametrize('code',[400,401,403,404,408,429,500,502,503])
def test_http_status_classification(code):
    response=httpx.Response(code,request=httpx.Request('GET','https://synthetic.test'))
    error=httpx.HTTPStatusError('不依赖语言',request=response.request,response=response)
    assert is_transient_error(error)==(code in (408,429) or code>=500)


def test_new_search_reference_is_not_progress_but_inventory_is():
    detector=LoopDetector()
    payload={'hits':[{'product_id':'P1001','skus':[{'sku_id':'P1001-S1','stock':2}]}],
             'query_conditions':{'price_max_major':300},'result_ref':'ctx_a','observed_at':'a'}
    original=deepcopy(payload)
    for i in range(3):
        current={**payload,'result_ref':f'ctx_{i}','observed_at':str(i)}
        hint=detector.observe('s','product_search_tool',{'normalized_query':'包'},[current],'success')
        assert bool(hint)==(i==2)
    assert payload==original
    changed=deepcopy(payload);changed['hits'][0]['skus'][0]['stock']=1
    assert detector.observe('s','product_search_tool',{'normalized_query':'包'},[changed],'success') is None
    assert detector.observe('s','product_search_tool',{'normalized_query':'另一个包'},[changed],'success') is None


def test_failure_requires_explicit_code_no_legacy_alias():
    from app.application.runtime import results
    with pytest.raises(ValueError):
        ToolResult('失败', state=ToolResultState.ERROR)
    assert not hasattr(results,'ToolChunk')


@pytest.mark.parametrize('code',['unavailable','internal'])
async def test_declared_execution_failure_counts_without_keyword(code):
    async def lookup():
        """明确执行故障。"""
        return ToolResult('暂时无法完成读取', state=ToolResultState.ERROR, error_code=code)
    registry=CircuitBreakerRegistry(failure_threshold=1)
    await call_tool(tool_graph(lookup,middlewares=[ToolResilienceMiddleware(registry)]))
    assert registry.status('lookup')=='open'


async def test_completed_tool_is_not_replayed_when_model_fails_then_checkpoint_resumes(monkeypatch):
    from contextvars import ContextVar
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langgraph.checkpoint.memory import InMemorySaver
    from tests.test_agent_handoff import ScriptedModel
    from app.application.runtime.tools import as_langchain_tool
    from app.infrastructure.eventbus import TradeEventBus
    class Model(ScriptedModel):
        broken: bool = True
        failures: int = 0
        def _generate(self,messages,stop=None,run_manager=None,**kwargs):
            if any(isinstance(m,ToolMessage) for m in messages):
                if self.broken:
                    self.failures+=1
                    raise httpx.ReadTimeout('synthetic')
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content='已有结果，无需再次执行'))])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[
                {'id':'once','name':'prepare','args':{}}]))])
    writes=[]
    async def prepare():
        """记录一个合成执行结果，不连接真实交易。"""
        writes.append(1);return ToolResult('准备完成，等待用户确认')
    model=Model()
    middleware=GatewayModelMiddleware(SimpleNamespace(llm_fallback_model='',llm_model='test',llm_max_retries=2),None, client=None)
    graph=create_agent(model,tools=[as_langchain_tool(prepare)],middleware=[middleware],checkpointer=InMemorySaver())
    orchestrator=MainAgentOrchestrator.__new__(MainAgentOrchestrator)
    orchestrator._bus=TradeEventBus()
    orchestrator._native_observer=ContextVar('test-observer',default=None)
    session=SimpleNamespace(graph=graph,config={'configurable':{'thread_id':'recovery'}})
    monkeypatch.setattr(asyncio,'sleep',AsyncMock())
    with pytest.raises(httpx.ReadTimeout):
        await orchestrator._reply('recovery',session,[HumanMessage(content='准备')])
    assert model.failures==3 and writes==[1]
    model.broken=False
    result=await graph.ainvoke(None,session.config)
    assert writes==[1] and result['messages'][-1].content=='已有结果，无需再次执行'
