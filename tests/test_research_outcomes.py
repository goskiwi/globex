"""研究成果的局限不等于执行失败，明确要求仍按来源核验。"""
from tests.test_agent_handoff import call,submission,factories,task,context

async def test_optional_unknown_is_delivered_without_research_failure(tmp_path,monkeypatch,context):
    _,model,dispatch,_=factories(tmp_path,monkeypatch,[call('read_candidates',{}),submission(candidates=[{'product_id':'P1001','sku_id':'P1001-S1','assessment':'根据已读资料比较','caveats':['目录未标注可选属性']}],caveats=['资料未涵盖可选信息'])])
    output=(await dispatch('search_agent',task())).data
    assert output['status']=='completed' and output['execution_status']=='completed'
    assert output['caveats']==['资料未涵盖可选信息'] and output['unmet_requirements']==[] and model.cursor==2

async def test_unmet_requirements_must_reference_buyer_scope(tmp_path,monkeypatch,context):
    valid={'product_id':'P1001','sku_id':'P1001-S1','assessment':'依据已读资料'}
    _,model,dispatch,_=factories(tmp_path,monkeypatch,[call('read_candidates',{}),submission(candidates=[valid],unmet_requirements=['研究者自行增加的条件']),submission(candidates=[valid],caveats=['可选资料未知'])])
    output=(await dispatch('search_agent',task())).data
    assert output['status']=='completed' and model.cursor==3 and not output['unmet_requirements']

async def test_explicit_unmet_requirement_remains_partial(tmp_path,monkeypatch,context):
    _,_,dispatch,_=factories(tmp_path,monkeypatch,[call('read_candidates',{}),submission(candidates=[{'product_id':'P1001','assessment':'没有对应证明'}],unmet_requirements=['必须明确提供防雨证明'])])
    output=(await dispatch('search_agent',task(requirements=['必须明确提供防雨证明']))).data
    assert output['status']=='partial' and output['unmet_requirements']==['必须明确提供防雨证明']

async def test_saved_report_is_presented_by_reference_without_dispatching_again(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel
    from tests.test_langgraph_runtime import _container
    from tests.test_task_execution import ask
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003','sku_id':'P1003-S1'}),submission(candidates=[{'product_id':'P1003','sku_id':'P1003-S1','assessment':'依据目录提供候选'}])])
    parent=ScriptedModel(responses=[call('update_shopping_state',{'update':{'plan':[{'id':'bag','goal':'比较背包候选'}]}}),call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag','goal':'比较背包候选'}}),call('present_research',{'reports': [{'task_id': task_id, 'title': task_id} for task_id in ['bag']]})])
    c=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        result,state=await ask(c,'请专家研究背包候选，只交回研究结果。')
        assert child.cursor==2 and parent.cursor==3 and result.status=='completed'
        assert result.presentations[-1]['kind']=='research' and len(state['task_plan']['tasks'])==1
    finally:await c.shutdown()



async def test_one_report_does_not_complete_two_goal_plan(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel
    from tests.test_langgraph_runtime import _container
    from tests.test_task_execution import ask
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003','sku_id':'P1003-S1'}),submission(candidates=[{'product_id':'P1003','sku_id':'P1003-S1','assessment':'已核验背包'}])])
    parent=ScriptedModel(responses=[call('update_shopping_state',{'update':{'plan':[{'id':'bag','goal':'研究背包'},{'id':'ear','goal':'研究耳机'}]}}),call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag','goal':'研究背包'}}),call('present_research',{'reports': [{'task_id': task_id, 'title': task_id} for task_id in ['bag']]})])
    c=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        result,state=await ask(c,'分别研究背包、耳机，交回两份研究。')
        assert result.status=='partial' and result.presentations[-1]['status']=='completed'
        assert result.presentations[-1]['payload']['plan_outcome']['remaining_count']==0
        assert '还有1项计划任务尚未完成' in result.final_text
        assert len(result.presentations[-1]['payload']['reports'])==1 and state['task_plan']['tasks']['bag']['delivery']=='research'
    finally:await c.shutdown()


async def test_one_catalog_read_carries_distinct_sku_quotes_and_canonical_research_facts():
    from types import SimpleNamespace
    from app.application.runtime.handoff import TaskEvidence
    from app.application.agents.handoff import ResearchSubmission
    from app.application.usecases.catalog_search import CatalogSearchUseCase
    from app.domain.catalog.product_search_spec import ProductSearchSpec
    from app.domain.catalog.product import Product
    from app.domain.catalog.sku import Sku
    from app.domain.catalog.money import Money
    from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
    item = Product(product_id='P9000', title='合成背包', brand='测试', category='旅行装备',
        description='合成目录资料', product_type='背包', origin_country='CN', ships_to=['CN'], skus=[
            Sku('P9000-S1', '黑色', Money.from_major_units(100, 'CNY'), 5),
            Sku('P9000-S2', '蓝色', Money.from_major_units(200, 'CNY'), 4)])
    result = await CatalogSearchUseCase(InMemoryProductRepository([item])).execute(
        ProductSearchSpec(product_id='P9000', ship_to='CN', target_currency='USD'))
    rows = result['hits'][0]['skus']
    assert [s['landed_price']['items'][0]['sku_id'] for s in rows] == ['P9000-S1', 'P9000-S2']
    assert [s['landed_price']['items'][0]['source_unit_price_minor'] for s in rows] == [10000, 20000]
    assert all(s['landed_price']['currency'] == 'USD' and s['landed_price']['ship_to'] == 'CN' for s in rows)
    evidence = TaskEvidence('s')
    evidence.capture(SimpleNamespace(shopping_session_id='s', type='tool.result', payload={
        'tool': 'product_search_tool', 'result_ref': 'ctx_catalog', 'observed_at': '2026-10-19T08:00:00+00:00', **result}))
    research = evidence.delivered(ResearchSubmission(candidates=[
        {'product_id': 'P9000', 'sku_id': s['sku_id'], 'assessment': '依据同次目录观察比较'} for s in rows]))
    for candidate in research.candidates:
        facts = candidate.facts
        assert len(facts['quotes']) == 1
        assert facts['quotes'][0]['line']['sku_id'] == candidate.sku_id
        assert facts['quotes'][0]['result_ref'] == facts['result_ref'] == 'ctx_catalog'
        assert facts['quotes'][0]['line']['quantity'] == 1
