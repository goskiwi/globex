"""只读审查反例：断言期望行为，用真实原生图/目录/临时 SQLite 验证，不改产品。"""
import json
import pytest
from langchain_core.messages import AIMessage,HumanMessage
from ag_ui.core import RunAgentInput
from app.application.agents.shopping_state import ShoppingWork,Filters,compile_search
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.tools.product_details_tool import build_product_details_tool
from app.application.tools.product_view_tool import build_product_view_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.intents import SubmitIntentInput
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.eventbus import TradeEventBus
from tests.test_langgraph_runtime import _container,ScriptedModel as ModeModel,PreferenceStore
from tests.test_agent_handoff import ScriptedModel,call
from app.application.memory.preference_selector import PreferenceSelector


def context():
    return ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY',
        effective_search=compile_search(ShoppingWork(filters=Filters(target_currency='CNY',ship_to='CN')))))


@pytest.mark.parametrize('product,sku',[('P40018','P40018-S2'),('P3004','P3004-S2')])
async def test_requested_sku_survives_read_to_display(tmp_path,product,sku):
    catalog=CatalogSearchUseCase(InMemoryProductRepository());bus=TradeEventBus();evidence=ContextEvidenceStore(tmp_path/'evidence.db');token=context()
    try:
        read=await build_product_details_tool(catalog,bus,evidence)(product,sku)
        shown=await build_product_view_tool(catalog,evidence,bus)([{'product_id':product,'sku_id':sku}],'查看这款商品的当前资料。')
        a=read.data['hits'][0];b=shown.data['hits'][0]
        print(json.dumps({'read_sku':a['default_sku_id'],'display_sku':b['default_sku_id'],'read_price':a['price_major'],'display_price':b['price_major'],'all_display_skus':[s['sku_id'] for s in b['skus']]},ensure_ascii=False))
        assert a['default_sku_id']==b['default_sku_id'],'明确读取S2后，展示重新默认成S1'
    finally:ShoppingContext.reset(token)


async def test_readonly_display_accepts_correct_identifier_in_explanation(tmp_path):
    catalog=CatalogSearchUseCase(InMemoryProductRepository());evidence=ContextEvidenceStore(tmp_path/'evidence.db');token=context()
    try:
        result=await build_product_view_tool(catalog,evidence,TradeEventBus())([{'product_id':'P1003'}],'查看 P1003 的当前规格和库存。')
        print(json.dumps({'ok':result.ok,'result':result.data},ensure_ascii=False))
        assert result.ok,'商品存在且只读查看，因为说明含编号被拒绝'
    finally:ShoppingContext.reset(token)


async def test_failed_independent_lookup_does_not_erase_successful_card(tmp_path,monkeypatch):
    calls=AIMessage(content='',tool_calls=[
        {'id':'show','name':'recommend_products','args':{'picks': [{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'quantity': 1, 'reason': '依据目录核对当前规格'}], 'guidance': '这款背包的目录资料已核对。','turn_action':'finish'}},
        {'id':'bad-order','name':'query_order_tool','args':{'order_id':'synthetic-missing-order'}}])
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003','sku_id':'P1003-S1'}),calls,AIMessage(content='背包已核对；该订单未找到。')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        await c.runs.startup()
        request=RunAgentInput.model_validate({'threadId':'s','runId':'audit-delivery','state':{},'tools':[],'context':[],
            'messages':[{'id':'u','role':'user','content':'推荐P1003-S1，同时查询synthetic-missing-order订单。'}],
            'forwardedProps':{'buyerId':'buyer'}})
        await c.runs.start(request,'buyer');run=await c.runs.wait('audit-delivery','buyer',5)
        records=await c.orchestrator._evidence_store.search('buyer','s',kind='recommendation')
        print(json.dumps({'saved_recommendations':len(records),'run_status':run['status'],'presentations':run['result']['presentations'],
              'visible_presentations':run['state'].get('presentations'),'model_calls':model.cursor},ensure_ascii=False))
        assert len(records)==1,'夹具必须实际成功推荐'
        assert any(p['kind']=='recommendation' for p in run['state'].get('presentations',[])),'无关查询失败使成功推荐不再发布卡片'
    finally:await c.shutdown()


async def test_pending_memory_does_not_discard_new_readonly_question(tmp_path,monkeypatch):
    store=PreferenceStore();model=ModeModel(mode='approval');c=await _container(tmp_path,monkeypatch,model)
    factory=c.orchestrator._sessions._main_factory;factory._preference_store=store;factory._preference_selector=PreferenceSelector()
    try:
        first=await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','记住我喜欢轻便设计'))
        before=len(model.seen);query='暂时不处理记忆审批，先只查P1003-S1库存。'
        result=await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY',query))
        graph=c.orchestrator._sessions._agents['s'];state=await graph.graph.aget_state(graph.config)
        saved=any(isinstance(m,HumanMessage) and m.text==query for m in state.values['messages'])
        print(json.dumps({'first_status':first.status,'second_status':result.status,'second_text':result.final_text,
              'new_model_calls':len(model.seen)-before,'new_question_in_graph':saved,'writes':len(store.items)},ensure_ascii=False))
        assert not store.items,'审批仍必须阻止记忆写入'
        assert not saved and result.status=='needs_input'
    finally:await c.shutdown()


async def test_saved_unfinished_plan_does_not_mark_independent_lookup_partial(tmp_path,monkeypatch):
    model=ScriptedModel(responses=[call('set_research_plan',{'plan': [{'id':'bag','goal':'研究背包'},{'id':'earphone','goal':'研究耳机'}]}),
        AIMessage(content='计划已保存，暂不执行。'),call('product_search_tool',{'product_id':'P1003','sku_id':'P1003-S1'}),AIMessage(content='已经核对这款背包当前库存。')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','保存背包、耳机两步研究计划，暂不执行。'))
        result=await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','先不开展计划，只查P1003-S1库存，保留原计划。'))
        agent=c.orchestrator._sessions._agents['s'];state=(await agent.graph.aget_state(agent.config)).values
        print(json.dumps({'lookup_status':result.status,'text':result.final_text,'old_steps':state['task_plan']['steps'],'model_calls':model.cursor},ensure_ascii=False))
        assert len(state['task_plan']['steps'])==2,'原计划必须保留'
        assert result.status=='completed','无关的未完成旧计划使库存查询被标为partial'
    finally:await c.shutdown()


def test_accepted_card_has_reply_anchor_before_run_finishes_and_after_failure():
    from tests.test_product_delivery import adapter
    from tests.presentation_helpers import accept
    target,events=adapter()
    accept(target,'recommendation',{'guidance':'已核验的候选见卡片','hits':[{'product_id':'P1003'}]})
    messages=next(e.messages for e in reversed(events) if e.type=='MESSAGES_SNAPSHOT')
    assert messages[-1].id==f'{target.request.run_id}:final:answer'
    assert not any(e.type in {'RUN_FINISHED','RUN_ERROR'} for e in events)
    target.fail('另一项工作失败')
    assert target.state['presentations'][0]['payload']['hits'][0]['product_id']=='P1003'
    assert events[-1].type=='RUN_ERROR'
