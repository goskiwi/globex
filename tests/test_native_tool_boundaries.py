"""原生工具节点的错误状态、半截流、取消与评测事件。"""
import asyncio
import pytest
from app.application.runtime.results import ToolResult, ToolResultState
from app.application.runtime.middleware import ToolResilienceMiddleware, BusinessToolMiddleware
from app.application.harness.loop_detector import LoopDetector
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from tests.native_tool_helpers import tool_graph, call_tool
from tests.test_queue_reliability import isolated_redis_url


@pytest.fixture(autouse=True)
def scope():
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)


async def test_business_error_status_survives_without_sentinel_or_harness():
    async def rejected():
        """返回业务拒绝。"""
        return ToolResult('目的国不支持', state=ToolResultState.ERROR, error_code="business_rejected")
    result=await call_tool(tool_graph(rejected,middlewares=[]))
    assert result.status=='error' and '目的国不支持' in result.content


@pytest.mark.parametrize('failure',['timeout','exception','error_result'])
async def test_partial_tool_stream_never_becomes_success_and_closes(failure):
    closed=asyncio.Event()
    async def partial():
        """失败前曾产生进度。"""
        try:
            yield ToolResult('不能当作最终成功的半截结果')
            if failure=='timeout':await asyncio.Event().wait()
            elif failure=='exception':raise RuntimeError('secret-provider-detail')
            else:yield ToolResult('503 Service Unavailable', state=ToolResultState.ERROR, error_code="unavailable")
        finally:
            closed.set()
    registry=CircuitBreakerRegistry(failure_threshold=1)
    graph=tool_graph(partial,middlewares=[ToolResilienceMiddleware(registry,timeouts={'partial':.05})])
    result=await call_tool(graph)
    assert closed.is_set() and result.status=='error'
    assert '半截结果' not in result.content and 'secret-provider-detail' not in result.content
    assert registry.status('partial')=='open'


async def test_caller_cancellation_closes_tool_and_is_not_infrastructure_failure():
    entered,closed=asyncio.Event(),asyncio.Event()
    async def waiting():
        """可取消工具。"""
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:closed.set()
    registry=CircuitBreakerRegistry(failure_threshold=1)
    task=asyncio.create_task(call_tool(tool_graph(waiting,middlewares=[ToolResilienceMiddleware(registry)])))
    try:
        await asyncio.wait_for(entered.wait(),2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert closed.is_set() and registry.status('waiting')=='closed'
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)


async def test_evaluation_observes_result_before_loop_notice_only_when_enabled():
    from app.infrastructure.context_usage import evaluation_evidence_sink
    samples=[]
    token=evaluation_evidence_sink.set(samples.append)
    async def lookup():
        """固定业务结果。"""
        return ToolResult({'hits':[]})
    graph=tool_graph(lookup,middlewares=[BusinessToolMiddleware(
        LoopDetector(repeat_threshold=2),TradeEventBus())])
    try:
        await call_tool(graph)
        result=await call_tool(graph)
        assert result.artifact['notices']
        observations=[s['payload'] for s in samples if s['kind']=='tool_result']
        assert len(observations)==2
        assert all(s['result']==[{'hits':[]}] and s['state']=='success' for s in observations)
        assert any(s['kind']=='tool_notice' for s in samples)
    finally:
        evaluation_evidence_sink.reset(token)


async def test_native_tools_share_circuit_between_independent_redis_clients(isolated_redis_url):
    from app.infrastructure.cache.redis_cache import RedisCache
    from app.infrastructure.shared_breaker import SharedCircuitBreakerRegistry
    caches=[RedisCache(isolated_redis_url),RedisCache(isolated_redis_url)]
    registries=[SharedCircuitBreakerRegistry(cache,failure_threshold=1,reset_seconds=60) for cache in caches]
    calls=[]
    async def search():
        """临时上游故障。"""
        calls.append(True)
        return ToolResult('503 Service Unavailable', state=ToolResultState.ERROR, error_code="unavailable")
    try:
        assert all([await cache.ping() for cache in caches])
        first=await call_tool(tool_graph(search,middlewares=[ToolResilienceMiddleware(registries[0])]))
        second=await call_tool(tool_graph(search,middlewares=[ToolResilienceMiddleware(registries[1])]))
        assert first.status==second.status=='error'
        assert '已熔断' in second.content and len(calls)==1
    finally:
        for cache in caches:await cache.close()
