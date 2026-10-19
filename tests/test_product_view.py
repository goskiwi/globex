"""展示本身不写状态；用户后续明确购买走确认单，原推荐条件不变。"""
import json
import pytest
from ag_ui.core import RunAgentInput

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.agents.shopping_state import ShoppingWork,Filters,compile_search
from app.application.tools.product_view_tool import build_product_view_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEvent
from app.infrastructure.ag_ui_journal import AGUIJournal
from tests.trade_test_helpers import confirmation_env
from tests.test_ag_ui_journal import body
from tests.trade_test_helpers import test_address as address
from app.application.usecases.order_usecases import OrderItemInput


async def test_blocked_products_are_displayed_without_becoming_candidates(confirmation_env):
    e=confirmation_env;catalog=CatalogSearchUseCase(e.products,pricing=e.service._pricing)
    work=ShoppingWork(filters=Filters(price_max_major=100,target_currency='CNY'))
    effective=compile_search(work)
    token=ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY',effective_search=effective))
    try:
        tool=build_product_view_tool(catalog,e.evidence,e.bus)
        result=await tool([{'product_id':'P3020'},{'product_id':'P3021'}],'下面按平台展示 Roamix 单侧支撑颈枕的完整资料。')
        assert result.ok
        payload=json.loads(result.text)
        assert payload['purpose']=='product_view' and len(payload['hits'])==2
        assert all(len(p['skus'])==2 for p in payload['hits'])
        assert all('over_price_cap' in s['constraint_issues'] for p in payload['hits'] for s in p['skus'])
        assert effective['parameters']['price_max_major']==100 and not work.selections
        assert await e.evidence.find_product('buyer','s',product_id='P3021',sku_id='P3021-S2') is None
        assert not (await tool([{'product_id':'P99999'}],'查看指定商品')).ok
    finally:ShoppingContext.reset(token)


async def test_view_then_explicit_purchase_keeps_budget_and_correct_platform_sku(confirmation_env):
    e=confirmation_env;catalog=CatalogSearchUseCase(e.products,pricing=e.service._pricing)
    work=ShoppingWork(filters=Filters(price_max_major=100,target_currency='CNY'))
    effective=compile_search(work)
    token=ShoppingContext.set(ShoppingContextSnapshot('view-purchase','buyer','zh-CN','CNY',effective_search=effective))
    try:
        result=await build_product_view_tool(catalog,e.evidence,e.bus)([{'product_id':'P3020'},{'product_id':'P3021'}],'查看 Roamix 颈枕。')
        assert result.ok and not work.selections
        stock=await e.store.get_inventory(['P3021-S2'])
        with pytest.raises(ValueError,match='库存不足'):
            await e.service.prepare_order('buyer','view-purchase',[OrderItemInput('P3020','P3020-S1',1)],address())
        pending=await e.service.prepare_order('buyer','view-purchase',[OrderItemInput('P3021','P3021-S2',1)],address(),'CNY')
        confirmation=pending['confirmation']
        assert confirmation['status']=='pending' and 'order' not in pending
        line=confirmation['payload']['items'][0]
        assert line['product_id']=='P3021' and line['sku_id']=='P3021-S2'
        assert line['unit_price_minor']==21399 and confirmation['payload']['total_amount_minor']>10000
        assert await e.store.get_inventory(['P3021-S2'])==stock
        resolved=await e.service.resolve(confirmation['confirmation_id'],'buyer','view-purchase',confirmation['snapshot_hash'],True)
        assert resolved['order']['status']=='CONFIRMED'
        assert resolved['confirmation']['payload']['items'][0]['sku_id']=='P3021-S2'
        assert (await e.store.get_inventory(['P3021-S2']))['P3021-S2']==stock['P3021-S2']-1
        assert work.filters.price_max_major==100 and effective['parameters']['price_max_major']==100
    finally:ShoppingContext.reset(token)


async def test_view_projects_selected_sku_without_hiding_other_choices(confirmation_env):
    e=confirmation_env
    selected=({'product_id':'P1003','sku_id':'P1003-S2','quantity':1},)
    context=ShoppingContextSnapshot('selected-view','buyer','zh-CN','CNY',
        effective_search={'parameters':{'target_currency':'CNY','ship_to':'CN'}},selected_lines=selected)
    token=ShoppingContext.set(context)
    try:
        tool=build_product_view_tool(CatalogSearchUseCase(e.products,pricing=e.service._pricing),e.evidence,e.bus)
        result=await tool([{'product_id':'P1003','sku_id':'P1003-S2'}],'查看雾霾蓝及其他颜色。')
        assert result.ok
        card=json.loads(result.text)['hits'][0]
        assert card['default_sku_id']=='P1003-S2'
        assert 'selected_sku_id' not in card
        assert {s['sku_id'] for s in card['skus']}=={'P1003-S1','P1003-S2'}
        assert card['landed_price']['items'][0]['sku_id']=='P1003-S2'
        assert ShoppingContext.current()==context
    finally:ShoppingContext.reset(token)


@pytest.mark.parametrize('complete',[False,True])
def test_view_commit_is_explicit_and_does_not_replace_recommendation(complete):
    from tests.presentation_helpers import accept
    target=AGUIRunAdapter(RunAgentInput.model_validate(body()),lambda event:None)
    accept(target,'recommendation',{'hits':[{'product_id':'previous'}]},identifier='before')
    if complete:accept(target,'product_view',{'purpose':'product_view','guidance':'查看资料','hits':[{'product_id':'P3020'}],'result_ref':'ctx_view'},identifier='view')
    target.finish('查看资料','completed',None)
    assert len(target.state['presentations'])==1+int(complete)
    assert target.state['presentations'][0]['kind']=='recommendation'



async def test_journal_restores_view_with_its_original_reply_after_followup(tmp_path):
    journal=AGUIJournal(tmp_path/'runs.db')
    await journal.reserve(body('view'),'b1','owner')
    payload={'purpose':'product_view','guidance':'详情资料','hits':[{'product_id':'P3020'}],
             'result_ref':'ctx_view','runId':'view'}
    await journal.append('view','owner',[
        {'type':'MESSAGES_SNAPSHOT','messages':[{'id':'view:final:answer','role':'assistant','content':'详情资料'}]},
        {'type':'STATE_SNAPSHOT','snapshot':{'presentations':[{'id':'view-record','kind':'product_view','status':'completed','runId':'view','payload':payload}]}},
        {'type':'RUN_FINISHED','runId':'view','threadId':'s1'}])
    await journal.reserve(body('followup'),'b1','owner')
    await journal.append('followup','owner',[{'type':'RUN_FINISHED','runId':'followup','threadId':'s1'}])
    restored=await journal.session('s1','b1')
    assert restored['presentations'][0]['payload']==payload
    assert restored['presentations'][0]['runId']=='view'
    assert restored['run']['state'].get('presentations',[])==[]


async def test_native_main_closes_with_view_and_keeps_budget(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput
    guidance='下面是 Roamix 单侧颈枕的资料，按平台分别展示。'
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{'filters':{'price_max_major':100,'target_currency':'CNY'}}}),
        call('get_product_details',{'product_id':'P3020'}),
        call('show_product_details',{'targets':[{'product_id':'P3020'},{'product_id':'P3021'}],'guidance':guidance})])
    container=await _container(tmp_path,monkeypatch,model)
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','商品价100元人民币，看看颈枕'))
        assert (result.presentations[-1]["kind"]=="product_view") and result.final_text==guidance and model.cursor==3
        state=await container.orchestrator._sessions._agents['s'].graph.aget_state({'configurable':{'thread_id':'s'}})
        assert state.values['shopping_work']['filters']['price_max_major']==100
        assert not await container.orchestrator._evidence_store.search('b','s',kind='display_batch')
    finally:await container.shutdown()


@pytest.mark.parametrize('product,sku',[('P3004','P3004-S2'),('P40018','P40018-S2')])
async def test_readonly_target_retains_requested_sku_without_purchase(tmp_path,product,sku):
    from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
    from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
    from app.infrastructure.eventbus import TradeEventBus
    from app.application.tools.product_details_tool import build_product_details_tool
    catalog=CatalogSearchUseCase(InMemoryProductRepository());store=ContextEvidenceStore(tmp_path/'evidence.db');bus=TradeEventBus()
    context=ShoppingContextSnapshot('view-target','buyer','zh-CN','CNY',effective_search={'parameters':{'target_currency':'CNY','ship_to':'CN'}})
    token=ShoppingContext.set(context)
    try:
        read=await build_product_details_tool(catalog,bus,store)(product,sku)
        shown=await build_product_view_tool(catalog,store,bus)([{'product_id':product,'sku_id':sku}],f'查看 {sku} 当前规格和库存。')
        assert shown.ok
        first=read.data['hits'][0];second=shown.data['hits'][0]
        assert first['default_sku_id']==second['default_sku_id']==sku
        assert first['price_major']==second['price_major']
        assert second['landed_price']['items'][0]['sku_id']==sku
        assert ShoppingContext.current()==context and not context.selected_lines
    finally:ShoppingContext.reset(token)
