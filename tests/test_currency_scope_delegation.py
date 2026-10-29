"""经真实主子图、需求修改和计划投影验证币种来源；不手工构造effective_search。"""
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.test_agent_handoff import ScriptedModel, call, submission
from tests.test_langgraph_runtime import _container
from tests.test_task_execution import ask


def child_model():
    return ScriptedModel(responses=[
        call('product_search_tool', {'product_id':'P1003', 'sku_id':'P1003-S1'}),
        submission(candidates=[{'product_id':'P1003', 'sku_id':'P1003-S1',
                                'evidence_fields':['description','weight_kg']}]),
    ])


@pytest.mark.parametrize('planned', [False, True], ids=['independent','planned'])
@pytest.mark.parametrize('parent_currency,local_filters,expected_currency,expected_constraint', [
    ('USD', None, 'USD', 'USD'),
    (None, {'target_currency':'USD','landed_budget_major':100}, 'USD', 'USD'),
    ('USD', {'target_currency':None}, 'CNY', None),
], ids=['inherit-parent','local-usd','local-clear'])
async def test_currency_source_reaches_actual_child_and_preserves_parent(
    tmp_path,monkeypatch,planned,parent_currency,local_filters,expected_currency,expected_constraint,
):
    parent_filters={'ship_to':'CN'}
    if parent_currency is not None:
        parent_filters.update(target_currency=parent_currency,landed_budget_major=100)
    responses=[call('set_shopping_requirements',{'changes':{'filters':parent_filters}})]
    if planned:
        step={'id':'bag','goal':'合成币种来源研究'}
        if local_filters is not None:step['filters']=local_filters
        responses.append(call('set_research_plan',{'plan':[step]}))
        task={'step_id':'bag'}
    else:
        task={'goal':'合成币种来源研究'}
        if local_filters is not None:task['filters']=local_filters
    responses += [call('task_dispatch',{'subagent_type':'search_agent','task':task}),
                  AIMessage(content='本轮合成研究已读取，未选购或交易。')]
    parent=ScriptedModel(responses=responses)
    child=child_model()
    container=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**k:child)
    try:
        _,state=await ask(container,'按父条件和子任务局部条件研究P1003-S1，不下单。')
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch')
        assert receipt.status=='success'
        data=receipt.artifact['data'];effective=data['scope']['effective_search']
        assert effective['parameters']['target_currency']==expected_currency
        assert effective['currency_constraint']==expected_constraint
        assert data['scope']['parent_conditions']['currency_constraint']==parent_currency
        assert data['candidates'][0]['facts']['sku']['display_currency']==expected_currency
        # 验证实际注入子模型的条件与派发回执相同，不只检查编译函数。
        injected=next(m for m in child.seen[0] if m.name=='delegated_task')
        assert json.loads(injected.content)['parent_context']['effective_search']==effective
        saved=state['shopping_work']['filters']
        assert saved['target_currency']==parent_currency
        assert saved['landed_budget_major']==(100 if parent_currency is not None else None)
        assert saved['ship_to']=='CN'
        if local_filters is not None and local_filters.get('target_currency','not-set') is None:
            assert effective['parameters']['landed_budget_major'] is None
            assert '到手预算币种未明确，尚未执行总价上限' in data['unmet_requirements']
            assert data['status']=='partial'
        else:
            assert effective['parameters']['landed_budget_major']==100
            assert data['status']=='completed'
        assert child.cursor==2
        assert not await container.trade_store.list_confirmations(buyer_id='buyer',session_id='s')
        assert (await container.trade_store.list_orders(buyer_id='buyer'))['total']==0
    finally:
        await container.shutdown()


async def test_explicit_default_currency_invalidates_completed_research_through_real_plan(
    tmp_path,monkeypatch,
):
    child=child_model()
    parent=ScriptedModel(responses=[
        call('set_shopping_requirements',{'changes':{'filters':{'ship_to':'CN'}}}),
        call('set_research_plan',{'plan':[{'id':'bag','goal':'合成默认币种研究'}]}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}),
        call('present_research',{'reports':[{'task_id':'bag','title':'合成币种研究'}]},turn_action='finish'),
        call('set_shopping_requirements',{'changes':{'filters':{'target_currency':'CNY'}}}),
        call('present_research',{'reports':[{'task_id':'bag','title':'尝试复用旧研究'}]},turn_action='finish'),
        AIMessage(content='币种现已明确，旧研究需要重新核验，未复用交付。'),
    ])
    container=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**k:child)
    try:
        first,state=await ask(container,'未限定币种，按默认显示研究P1003-S1并交付报告，不下单。')
        original=deepcopy(state['task_plan']['tasks']['bag'])
        assert first.status=='completed' and original['status']=='completed' and not original['stale']
        assert original['scope']['parent_conditions']['currency_constraint'] is None
        assert original['scope']['parent_conditions']['parameters']['target_currency']=='CNY'
        assert len(first.presentations)==1 and first.presentations[0]['kind']=='research'

        second,state=await ask(container,'现在明确要求人民币口径，尝试查看同一研究。')
        current=state['task_plan']['tasks']['bag']
        assert state['shopping_work']['filters']['target_currency']=='CNY'
        assert current['stale'] is True
        assert current['scope']['parent_conditions']==original['scope']['parent_conditions']
        assert next(step for step in state['task_plan']['steps'] if step['id']=='bag')['status']=='pending'
        rejected=[m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='present_research'][-1]
        assert rejected.status=='error' and '已失效' in str(rejected.content)
        assert not second.presentations
        assert child.cursor==2, '状态变化不能隐式重跑子任务'
    finally:
        await container.shutdown()
