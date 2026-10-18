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
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
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
            'status': 'partial', 'issues': ['候选不足']}
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
    decision = {'candidates': [{'product_id': 'P1001', 'sku_id': 'P1001-S1', 'evidence_fields': ['description'],
        'facts': {'result_ref': 'ctx_source', 'historical': False, 'observed_at': 'now',
                  'product': {'description': '必须保留的长资料' * 10000}}}]}
    before = json.dumps(decision, ensure_ascii=False)
    projected = await bounded_tool_view(decision, None, None, kind='handoff', token_limit=2000)
    assert projected['candidates'][0]['evidence_fields'] == ['description']
    assert projected['candidates'][0]['facts']['offloaded'] is True
    assert projected['candidates'][0]['facts']['result_ref'] == 'ctx_source'
    assert json.dumps(decision, ensure_ascii=False) == before


def research_receipt(identifier, *, description='目录明确记载防水卷口'):
    """合成研究含两种数量/币种报价，不能因压缩混用。"""
    facts = {'result_ref': 'ctx_' + identifier, 'historical': False, 'observed_at': '2026-10-18',
        'product': {'product_id': identifier, 'title': identifier + ' 正确商品名', 'brand': '真实品牌',
                    'weight_kg': .38, 'description': description,
                    'dimensions_cm': {'length': 50}, 'rating_summary': {'average': 4.6, 'review_count': 217}},
        'sku': {'sku_id': identifier + '-S1', 'spec': '雾灰', 'stock': 8, 'price_major': 129, 'currency': 'CNY'},
        'quotes': [{'ship_to': 'CN', 'currency': currency, 'historical': False, 'observed_at': '2026-10-18',
                    'result_ref': 'ctx_' + identifier,
                    'line': {'product_id': identifier, 'sku_id': identifier + '-S1',
                             'quantity': quantity, 'total_amount_minor': amount, 'currency': currency}}
                   for quantity, currency, amount in [(1, 'CNY', 15400), (2, 'USD', 3800)]]}
    return {'agent': 'search_agent', 'status': 'completed', 'summary': '该品类研究完成',
        'scope': {'task_id': identifier, 'goal': '通勤研究', 'effective_search': {'parameters': {'ship_to': 'CN', 'target_currency': 'CNY'}},
                  'parent_conditions': {'unused': '原始条件' * 100}},
        'candidates': [{'product_id': identifier, 'sku_id': identifier + '-S1', 'evidence_fields': ['description', 'dimensions_cm'],
                        'open_questions': ['是否有电脑仓？'], 'unmet_constraints': [],
                        'facts': facts}],
        'evidence_refs': ['ctx_' + identifier], 'observed_products': ['仅观察未选中' * 200],
        'transaction_state': 'none', 'stop_reason': None, 'unmet_constraints': [], 'questions': [], 'issues': []}


async def test_current_research_keeps_facts_under_pressure_and_old_turn_can_archive(tmp_path):
    from app.application.runtime.results import project_data, message_data
    from tests.native_context_helpers import apply
    receipts = [research_receipt('P1003'), research_receipt('P1018')]
    messages = [HumanMessage(id='buyer-turn', name='buyer', content='两份研究，暂不最终推荐'),
        AIMessage(id='dispatch', content='', tool_calls=[
            {'id': str(i), 'name': 'task_dispatch', 'args': {}} for i in range(2)])]
    for i, receipt in enumerate(receipts):
        messages.append(project_data(ToolMessage(id=f'r{i}', tool_call_id=str(i), name='task_dispatch',
                                                content=''), receipt))
    state = {'messages': messages, 'read_tool_messages': ['r0', 'r1']}
    middleware = policy(tmp_path, product_tokens=1)
    middleware.model = None
    update = await middleware.compact_checkpoint(state, force=True)
    current = apply(state, update)
    for i, original in enumerate(receipts):
        receipt = next(m for m in current['messages'] if m.id == f'r{i}')
        shown = message_data(receipt)
        assert not shown.get('archived')
        assert shown['candidates'] == original['candidates']
        assert shown['candidates'][0]['open_questions'] == ['是否有电脑仓？']
        assert shown['transaction_state'] == 'none'
        saved = await middleware.store.get('buyer', 'handoff', shown['result_ref'])
        assert saved['data'] == original
        assert receipt.artifact['data'] == original
    assert not any('compact_view' in r for r in receipts), '原始证据不得被原地修改'
    # 新需求开始后，旧研究可按原规则回收，不永久保护全部历史。
    current['messages'].append(HumanMessage(id='new-turn', name='buyer', content='新任务，查水杯'))
    archived = await middleware.compact_checkpoint(current, force=True)
    assert all(message_data(m).get('archived') for m in archived['messages'])
    assert len(archived['messages']) == 2


async def test_first_large_handoff_retains_identity_sku_and_quote_conditions():
    from app.application.runtime.tool_view import bounded_tool_view
    original = research_receipt('P1003', description='必须回查的长目录段落' * 10000)
    before = json.dumps(original, ensure_ascii=False)
    view = await bounded_tool_view(original, None, None, kind='handoff', token_limit=2000)
    facts = view['candidates'][0]['facts']
    assert facts['product']['title'] == 'P1003 正确商品名'
    assert facts['product']['brand'] == '真实品牌' and facts['product']['weight_kg'] == .38
    assert facts['product']['rating_summary']['review_count'] == 217
    assert facts['sku'] == original['candidates'][0]['facts']['sku']
    assert facts['quotes'] == original['candidates'][0]['facts']['quotes']
    assert facts['offloaded'] and facts['omitted_product_fields'] == ['description']
    assert 'description' not in facts['product']
    assert view['candidates'][0]['open_questions'] == ['是否有电脑仓？']
    assert json.dumps(original, ensure_ascii=False) == before


async def test_protected_research_still_obeys_complete_request_capacity(tmp_path):
    from app.application.runtime.errors import ContextCapacityError
    from app.application.runtime.results import project_data
    receipt = research_receipt('P1003', description='当前待交付研究证据' * 100000)
    message = project_data(ToolMessage(id='large', tool_call_id='call', name='task_dispatch', content=''), receipt)
    middleware = policy(tmp_path, product_tokens=1)
    middleware.model = None
    with pytest.raises(ContextCapacityError):
        await middleware.compact_checkpoint({'messages': [
            HumanMessage(name='buyer', content='当前研究'), message], 'read_tool_messages': ['large']})


@pytest.mark.parametrize('count', [6, 14])
async def test_current_finalist_details_stay_available_with_existing_candidate_limit(tmp_path, count):
    from app.application.tools.recommendation_tools import MAX_DECISION_ITEMS
    state = history(count)
    state['messages'] = [m.model_copy(update={'name': 'get_product_details'}) if isinstance(m, ToolMessage) else m
                         for m in state['messages'] if not isinstance(m, HumanMessage)]
    state['messages'].insert(0, HumanMessage(id='current', name='buyer', content='核对本轮候选详情'))
    middleware = policy(tmp_path, product_tokens=1)
    middleware.model = None
    update = await middleware.compact_checkpoint(state)
    assert len(update['messages']) == max(0, count - MAX_DECISION_ITEMS)
    for message in update['messages']:
        assert json.loads(message.content)['archived']
    state['messages'].append(HumanMessage(id='next', name='buyer', content='新的查询'))
    old = await middleware.compact_checkpoint(state)
    assert len(old['messages']) == count - 2, '旧轮次恢复原有回收规则，不永久保护详情'


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
