"""交付职责边界：内部任务不作标题，推荐与组合预算互不混淆。"""
import json
from app.application.tools.recommendation_tools import build_recommendation_tool,build_bundle_tool,Pick
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.domain.catalog.product_search_spec import ProductSearchSpec
from tests.trade_test_helpers import confirmation_env


async def test_individual_choices_and_bundle_share_facts_but_apply_distinct_total_budget(confirmation_env):
    env=confirmation_env;catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    for pid in ['P1001','P1003']:
        await env.evidence.save('buyer-1','session-1','products',await catalog.execute(ProductSearchSpec(product_id=pid)))
    token=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY',effective_search={
        'parameters':{'ship_to':'CN','target_currency':'CNY','landed_budget_major':300}}))
    selected=[Pick(product_id=pid,sku_id=pid+'-S1',quantity=1,reason='符合本次用途') for pid in ['P1001','P1003']]
    try:
        individual=await build_recommendation_tool(catalog,env.evidence,env.bus)(selected,'分别考虑这两种选择')
        assert individual.ok and individual.data['mode']=='individual' and individual.data['quote'] is None
        assert not {'preferred_sku_id','dimensions','max_items'} & individual.data.keys()
        bundle=await build_bundle_tool(catalog,env.evidence,env.bus)(selected,'一起购买这两件')
        assert not bundle.ok and '超过当前预算' in bundle.text
        context=ShoppingContext.current()
        from dataclasses import replace
        ShoppingContext.set(replace(context,effective_search={'parameters':{'ship_to':'CN','target_currency':'CNY','landed_budget_major':400}}))
        bundle=await build_bundle_tool(catalog,env.evidence,env.bus)(selected,'一起购买这两件')
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
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),submission(candidates=[{
        'product_id':'P1003','sku_id':'P1003-S1','assessment':'根据已读目录比较收纳需求'}])])
    main=ScriptedModel(responses=[
        call('update_shopping_state',{'update':{'plan':[{'id':'bag','goal':internal}]}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag','goal':internal}}),
        call('present_research',{'reports':[{'task_id':'bag','title':title}]})])
    c=await _container(tmp_path,monkeypatch,main)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**k:child)
    try:
        result,state=await ask(c,'请专家研究背包，仅交研究报告')
        assert result.status=='completed' and main.cursor==3
        payload=result.presentations[0]['payload'];report=payload['reports'][0]
        assert report['title']==title and 'goal' not in report
        assert internal not in json.dumps(payload,ensure_ascii=False) and 'description/highlights' not in result.final_text
        assert title in result.final_text
        assert state['task_plan']['tasks']['bag']['scope']['goal']==internal
    finally:await c.shutdown()
