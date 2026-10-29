from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
"""交付职责边界：内部任务不作标题，推荐与组合预算互不混淆。"""
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
from langchain_core.messages import ToolMessage
from app.application.tools.recommendation_tools import build_recommendation_tool,build_bundle_tool,Pick
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.domain.catalog.product_search_spec import ProductSearchSpec
from tests.trade_test_helpers import confirmation_env


@pytest.mark.parametrize('status', ['completed', 'partial', 'needs_input', 'failed'])
async def test_research_receipt_is_brief_and_report_keeps_all_completion_details(status):
    from app.application.tools.research_delivery_tool import build_research_delivery_tool
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from tests.test_research_delivery_boundaries import research_receipt
    receipt = research_receipt('P1003')
    receipt['scope']['task_id'] = 'bag'
    receipt.update(status=status, caveats=['目录未标明背板结构'],
                   unmet_requirements=['独立电脑仓缺少证明'] if status == 'partial' else [],
                   questions=['需要放多大电脑？'] if status == 'needs_input' else [],
                   issues=['补充来源暂不可用'] if status == 'failed' else [],
                   transaction_state='awaiting_confirmation' if status == 'needs_input' else 'none')
    original = deepcopy(receipt)
    runtime = SimpleNamespace(state={'messages': [ToolMessage(content='', name='task_dispatch',
        tool_call_id='research', artifact={'data': receipt})], 'task_plan': {'tasks': {'bag': {'stale': False, 'scope': receipt['scope']}}}})
    result = await build_research_delivery_tool(ExchangeRateTable())(
        [{'task_id': 'bag', 'title': '通勤背包研究'}], runtime)
    assert result.ok
    assert result.data['guidance'] == '已交付1份研究报告。'
    report = result.data['reports'][0]
    for field in ('unmet_requirements', 'questions', 'issues', 'transaction_state', 'status'):
        assert report[field] == original[field]
    assert report['items'][0]['facts'] == original['candidates'][0]['facts']
    assert report['items'][0]['evidence_fields'] == original['candidates'][0]['evidence_fields']
    assert report['caveats'] != original['caveats']
    assert receipt == original


async def test_individual_choices_and_bundle_share_facts_but_apply_distinct_total_budget(confirmation_env):
    env=confirmation_env;catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    for pid in ['P1001','P1003']:
        await env.evidence.save('buyer-1','session-1','products',await catalog.execute(ProductSearchSpec(product_id=pid)))
    token=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY',effective_search=compile_search(ShoppingWork(filters=Filters(**{'ship_to': 'CN', 'target_currency': 'CNY', 'landed_budget_major': 300})), currency='CNY')))
    selected=[Pick(product_id=pid, sku_id=pid + '-S1', quantity=1) for pid in ['P1001','P1003']]
    try:
        individual=await build_recommendation_tool(catalog,env.evidence,env.bus)(selected,[])
        assert individual.ok and individual.data['mode']=='individual' and individual.data['quote'] is None
        assert not {'preferred_sku_id','dimensions','max_items'} & individual.data.keys()
        bundle=await build_bundle_tool(catalog,env.evidence,env.bus)(selected,[],'CN','CNY',None)
        assert not bundle.ok and '超过当前预算' in bundle.text
        context=ShoppingContext.current()
        from dataclasses import replace
        ShoppingContext.set(replace(context,effective_search=compile_search(ShoppingWork(filters=Filters(**{'ship_to': 'CN', 'target_currency': 'CNY', 'landed_budget_major': 400})), currency='CNY')))
        bundle=await build_bundle_tool(catalog,env.evidence,env.bus)(selected,[],'CN','CNY',None)
        assert bundle.ok and bundle.data['quote']['total_amount_minor']==36800
        assert [h['default_sku_id'] for h in bundle.data['hits']]==['P1001-S1','P1003-S1']
        assert await env.store.list_confirmations(buyer_id='buyer-1',session_id='session-1')==[]
    finally:ShoppingContext.reset(token)


async def test_main_supplies_public_title_without_copying_internal_goal(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call,submission
    from tests.test_langgraph_runtime import _container
    from tests.test_task_execution import ask
    internal='内部执行指令：检查description/highlights与SKU来源，逐项记录不可推断的结构；'*3
    title='通勤背包研究'
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),submission(candidates=[{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'evidence_fields': ['description', 'weight_kg']}])])
    main=ScriptedModel(responses=[
        call('set_research_plan',{'plan': [{'id':'bag','goal':internal}]}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}),
        call('present_research',{'reports':[{'task_id':'bag','title':title}]}, turn_action='finish')])
    c=await _container(tmp_path,monkeypatch,main)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**k:child)
    try:
        result,state=await ask(c,'请专家研究背包，仅交研究报告')
        assert result.status=='completed' and main.cursor==3
        payload=result.presentations[0]['payload'];report=payload['reports'][0]
        assert report['title']==title and 'goal' not in report
        assert internal not in json.dumps(payload,ensure_ascii=False) and 'description/highlights' not in result.final_text
        assert result.final_text == '已交付1份研究报告。'
        assert title not in result.final_text
        assert state['task_plan']['tasks']['bag']['scope']['goal']==internal
    finally:await c.shutdown()
