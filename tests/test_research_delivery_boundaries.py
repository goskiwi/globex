"""研究链路反例：证据不冒充选择，当前轮可整理，收尾仍可交付。"""
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.application.agents.handoff import build_submission_tool
from app.application.runtime.execution import ExecutionMiddleware
from app.application.runtime.handoff import TaskEvidence, HandoffContext, HandoffMiddleware
from app.application.runtime.delivery import FinalDeliveryMiddleware
from app.application.runtime.results import ToolResult
from app.application.runtime.tools import as_langchain_tool
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from tests.native_context_helpers import history, policy
from tests.native_model_helpers import client_model, completion
from tests.test_execution_stop import root


@pytest.fixture(autouse=True)
def scope():
    token = ShoppingContext.set(ShoppingContextSnapshot('handoff', 'buyer', 'zh-CN', 'CNY'))
    yield
    ShoppingContext.reset(token)


def test_272_observations_are_not_272_selected_candidates():
    evidence = TaskEvidence('handoff')
    evidence.capture(SimpleNamespace(shopping_session_id='handoff', type='tool.result', payload={
        'tool': 'product_search_tool', 'result_ref': 'ctx_original', 'hits': [
            {'product_id': f'P{i+1000}', 'description': '完整来源资料' * 100,
             'skus': [{'sku_id': f'P{i+1000}-S1'}]} for i in range(272)]}))
    stopped = evidence.stopped_result('context_capacity').model_dump()
    assert stopped['status'] == 'partial'
    assert stopped['candidates'] == []
    assert stopped['observed_product_count'] == 272
    assert stopped['evidence_refs'] == ['ctx_original']
    assert len(json.dumps(stopped, ensure_ascii=False)) < 2000
    assert len(evidence.products) == 272, '完整证据索引不能因模型投影缩小而删除'


@pytest.mark.parametrize('delegated', [False, True])
async def test_read_results_in_current_turn_can_be_archived(tmp_path, delegated):
    state = history(8)
    state['messages'] = [m for m in state['messages'] if not isinstance(m, HumanMessage)]
    state['messages'].insert(0, HumanMessage(id='current', name='delegated_task' if delegated else 'buyer',
                                           content='本次研究目标'))
    middleware = policy(tmp_path, product_tokens=1)
    middleware.model = None
    update = await middleware.compact_checkpoint(state)
    assert {m.id for m in update['messages']} == {f't{i}' for i in range(6)}
    assert update['context_statistics']['archived_result_count'] == 6
    for message in update['messages']:
        payload = json.loads(message.content)
        saved = await middleware.store.get('buyer', 'handoff', payload['result_ref'])
        assert saved['data']['hits']
    assert not any('archived' in m.content for m in state['messages'] if isinstance(m, ToolMessage))


@pytest.mark.parametrize('main', [False, True])
async def test_last_round_keeps_only_role_delivery_tools(tmp_path, main):
    requests, reads = [], []
    evidence = TaskEvidence('handoff')
    @tool
    def read():
        """读取一次合成事实。"""
        reads.append(1)
        evidence.successful_tools.add('read')
        return '已读取资料'
    async def recommend_products():
        """提交已有选择，不执行交易。"""
        return ToolResult({'guidance': '已有候选满足用途，缺少的第四款尚未确认。', 'hits': []})
    delivery = as_langchain_tool(recommend_products) if main else build_submission_tool()
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        name = 'read' if len(requests) == 1 else delivery.name
        args = {} if main or len(requests) == 1 else {
            'status': 'partial', 'summary': '已有资料，尚未筛选出四款', 'issues': ['候选不足']}
        response = completion()
        response['choices'][0].update(message={'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': f'c{len(requests)}', 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(args)}}]}, finish_reason='tool_calls')
        return httpx.Response(200, json=response)
    model = await client_model(tmp_path, handler)
    try:
        execution = ExecutionMiddleware(2, main=main, delivery_tools=(delivery.name,))
        middleware = [execution, FinalDeliveryMiddleware()] if main else [HandoffMiddleware(), execution]
        graph = create_agent(model, tools=[read, delivery], middleware=middleware,
                             context_schema=None if main else HandoffContext)
        result = await graph.ainvoke({'messages': [HumanMessage(content='研究')]},
                                    context=None if main else HandoffContext(evidence))
        assert reads == [1] and len(requests) == 2
        assert [t['function']['name'] for t in requests[1]['tools']] == [delivery.name]
        assert result['execution_stop'] == 'model_call_limit'
        if main:
            assert result['product_delivery_complete'] is True
            assert result['messages'][-1].content.startswith('已有候选')
        else:
            assert result['handoff_result']['status'] == 'partial'
    finally:
        await model.aclose()


async def test_root_capacity_stop_keeps_evidence_and_closes_checkpoint():
    from tests.test_agent_handoff import ScriptedModel, call
    from app.application.runtime.errors import ContextCapacityError
    bus = TradeEventBus()
    async def read():
        """先取得真实工具回执。"""
        bus.publish('handoff', 'tool.result', {'tool': 'product_search_tool',
            'result_ref': 'ctx_original', 'hits': [{'product_id': 'P1001', 'skus': []}]})
        return ToolResult('资料已保存')
    class CapacityModel(ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.cursor:
                raise ContextCapacityError('受保护输入超限')
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    graph = create_agent(CapacityModel(responses=[call('read', {})]), tools=[as_langchain_tool(read)],
                         checkpointer=InMemorySaver())
    runner, session = root(graph, bus)
    result = await runner._reply('handoff', session, [HumanMessage(content='研究')])
    assert result.status == 'partial' and result.stop_reason == 'context_capacity'
    assert 'ctx_original' in result.text
    assert (await graph.aget_state(session.config)).next == ()


async def test_closing_form_is_needs_input_not_failed():
    from tests.test_agent_handoff import ScriptedModel, call
    async def read():
        """读取已知信息。"""
        return ToolResult('仍缺用途')
    async def show_shopping_form():
        """已创建并保存问题。"""
        return ToolResult({'form_id':'saved-form','status':'awaiting_input'})
    model=ScriptedModel(responses=[call('read',{}),call('show_shopping_form',{})])
    graph=create_agent(model,tools=[as_langchain_tool(read),as_langchain_tool(show_shopping_form)],
        middleware=[ExecutionMiddleware(2,main=True,delivery_tools=('show_shopping_form',)),
                    FinalDeliveryMiddleware()],checkpointer=InMemorySaver())
    runner,session=root(graph,TradeEventBus())
    output=await runner._reply('handoff',session,[HumanMessage(content='需要补充用途')])
    assert output.status=='needs_input' and not output.product_delivery_complete
    assert model.cursor==2 and (await graph.aget_state(session.config)).next==()


def test_completed_status_alone_cannot_publish_an_unfinished_batch():
    from tests.test_product_delivery import adapter,emit
    target,_=adapter()
    emit(target,'recommendation.result',{'hits':[{'product_id':'P1001'}]})
    target.finish('完成','completed',None)
    assert target.state['recommendation'] is None


async def test_large_selected_facts_use_source_refs_without_truncation():
    from app.application.runtime.tool_view import bounded_tool_view
    decision = {'candidates': [{'product_id': 'P1001', 'sku_id': 'P1001-S1', 'reason': '已筛选',
        'facts': {'result_ref': 'ctx_source', 'historical': False, 'observed_at': 'now',
                  'product': {'description': '必须保留的长资料' * 10000}}}]}
    before = json.dumps(decision, ensure_ascii=False)
    projected = await bounded_tool_view(decision, None, None, kind='handoff', token_limit=2000)
    assert projected['candidates'][0]['reason'] == '已筛选'
    assert projected['candidates'][0]['facts']['offloaded'] is True
    assert projected['candidates'][0]['facts']['result_ref'] == 'ctx_source'
    assert json.dumps(decision, ensure_ascii=False) == before


@pytest.mark.parametrize('completed', [False, True])
def test_partial_run_only_publishes_a_completed_delivery(completed):
    from tests.test_product_delivery import adapter, emit
    target, _ = adapter()
    emit(target, 'recommendation.result', {'hits': [{'product_id': 'P1001'}]})
    target.finish('研究未完整完成，已有选择见卡片。', 'partial', 'model_call_limit',
                  product_delivery_complete=completed)
    assert bool(target.state['recommendation']) is completed


async def test_model_that_never_submits_cannot_keep_researching():
    from tests.test_agent_handoff import ScriptedModel, call
    bus, executed = TradeEventBus(), []
    async def read():
        """不断返回同一份证据。"""
        executed.append(1)
        bus.publish('handoff', 'tool.result', {'tool': 'product_search_tool',
            'result_ref': 'ctx_original', 'hits': [{'product_id': 'P1001', 'skus': []}]})
        return ToolResult('已保存')
    model = ScriptedModel(responses=[call('read', {}) for _ in range(10)])
    graph = create_agent(model, tools=[as_langchain_tool(read)],
        middleware=[ExecutionMiddleware(3, main=True, delivery_tools=('recommend_products',))],
        checkpointer=InMemorySaver())
    runner, session = root(graph, bus)
    output = await runner._reply('handoff', session, [HumanMessage(content='继续研究')])
    assert len(executed) == 2 and model.cursor == 3
    assert output.status == 'partial' and output.stop_reason == 'model_call_limit'
    assert not output.product_delivery_complete
    assert 'ctx_original' in output.text
    snapshot = await graph.aget_state(session.config)
    assert snapshot.next == ()
    calls = {c['id'] for m in snapshot.values['messages'] if isinstance(m, AIMessage) for c in m.tool_calls}
    assert calls == {m.tool_call_id for m in snapshot.values['messages'] if isinstance(m, ToolMessage)}


def test_original_failed_trace_returns_only_evidence_on_stop():
    from pathlib import Path
    results = json.loads((Path(__file__).parent /
        'fixtures/context-stop-search-results.json').read_text())
    evidence = TaskEvidence('handoff')
    for payload in results:
        evidence.capture(SimpleNamespace(shopping_session_id='handoff', type='tool.result', payload=payload))
    result = evidence.stopped_result('context_capacity')
    assert result.observed_product_count > 100
    assert result.candidates == [] and result.evidence_refs
    assert len(result.model_dump_json()) < 10000
