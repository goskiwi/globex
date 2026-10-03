"""详情读取与候选筛选分离；使用原目录、证据库和库存，不改购物条件。"""
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.application.agents.shopping_state import ShoppingWork,Filters,compile_search
from app.application.tools.product_details_tool import ProductDetailsInput,build_product_details_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.recommendation_tools import build_recommendation_tool,Pick
from app.application.tools.order_tools import _verified_order_items
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.context import ShoppingContext,ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env


def test_detail_contract_has_no_budget_override_and_rejects_mismatched_sku():
    assert ProductDetailsInput(product_id='P3020').sku_id is None
    with pytest.raises(ValidationError):ProductDetailsInput(product_id='P3020',price_max_major=None)
    with pytest.raises(ValidationError):ProductDetailsInput(product_id='P3020',sku_id='P3021-S1')


async def test_complete_out_of_budget_and_out_of_stock_details_do_not_change_filters(confirmation_env):
    e=confirmation_env
    catalog=CatalogSearchUseCase(e.products,pricing=e.service._pricing)
    work=ShoppingWork(filters=Filters(price_max_major=100,target_currency='CNY'),
                      excluded_skus=['P3021-S1'])
    effective=compile_search(work);before=deepcopy(effective)
    token=ShoppingContext.set(ShoppingContextSnapshot('detail-test','buyer','zh-CN','CNY',effective_search=effective))
    try:
        details=build_product_details_tool(catalog,e.bus,e.evidence)
        search=build_product_search_tool(catalog,e.bus,e.evidence)
        for product_id,platform in [('P3020','amazon'),('P3021','ebay')]:
            returned=await details(product_id)
            assert returned.ok
            data=json.loads(returned.text);p=data['hits'][0]
            assert data['existence_checked'] and not data['missing_identifiers']
            assert p['source_platform']==platform and len(p['skus'])==2
            assert p['description'] and p['dimensions_cm'] and p['weight_kg']==0.15
            assert all('over_price_cap' in s['constraint_issues'] for s in p['skus'])
            if product_id=='P3020':
                assert all(s['stock']==0 and 'out_of_stock' in s['constraint_issues'] for s in p['skus'])
            else:
                assert p['skus'][1]['display_price_major']==213.99
                assert p['skus'][1]['currency']=='USD' and p['skus'][1]['stock']>0
                assert p['skus'][0]['sku_id']=='P3021-S1' # 被排除规格仍可查看。
            blocked=json.loads((await search(product_id=product_id)).text)
            assert blocked['hits']==[] and blocked['filtered_out']
        assert effective==before and work.filters.price_max_major==100
        assert await e.evidence.find_product('buyer','detail-test',product_id='P3021',sku_id='P3021-S2') is None
        with pytest.raises(ValueError):
            await _verified_order_items(e.evidence,'buyer','detail-test',
                [{'product_id':'P3021','sku_id':'P3021-S2','quantity':1}])
        refused=await build_recommendation_tool(catalog,e.evidence,e.bus)(
            [Pick(product_id='P3021',sku_id='P3021-S2',quantity=1,reason='想查看这款')],
            'alternatives','P3021-S2',[], '这款只供查看。')
        assert not refused.ok
        assert len(await e.evidence.search('buyer','detail-test',kind='product_details'))==2
    finally:ShoppingContext.reset(token)


async def test_native_main_reads_both_platform_records_without_budget_change(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call
    from tests.test_langgraph_runtime import _container
    from app.application.agents.orchestrator import SubmitIntentInput
    from langchain_core.messages import AIMessage,ToolMessage
    reads=AIMessage(content='',tool_calls=[{'id':pid,'name':'get_product_details','args':{'product_id':pid}}
        for pid in ('P3020','P3021')])
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{
        'filters':{'price_max_major':100,'target_currency':'CNY'}}}),
        reads,AIMessage(content='这款仍能查看；两个平台记录分别缺货或超预算，预算保持不变。')])
    container=await _container(tmp_path,monkeypatch,model)
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY',
            '商品价100元人民币，查看Roamix航空颈枕'))
        assert result.status=='completed' and not result.product_delivery_complete
        agent=container.orchestrator._sessions._agents['s']
        state=await agent.graph.aget_state(agent.config)
        assert state.values['shopping_work']['filters']['price_max_major']==100
        receipts=[json.loads(m.content) for m in model.seen[-1] if isinstance(m,ToolMessage) and m.name=='get_product_details']
        assert len(receipts)==2 and all(len(p['hits'][0]['skus'])==2 for p in receipts)
        assert {p['hits'][0]['source_platform'] for p in receipts}=={'amazon','ebay'}
        assert not await container.orchestrator._evidence_store.search('b','s',kind='display_batch')
    finally:await container.shutdown()


def test_repeated_detail_reads_ignore_new_refs_but_notice_stock_changes():
    from app.application.harness.loop_detector import LoopDetector
    detector=LoopDetector()
    def result(index,stock):return {'hits':[{'product_id':'P3020','skus':[{'sku_id':'P3020-S1','stock':stock}]}],
                                  'result_ref':f'new-{index}','observed_at':index}
    for index in (1,2):assert detector.observe('s','get_product_details',{'product_id':'P3020'},result(index,0),'success') is None
    assert detector.observe('s','get_product_details',{'product_id':'P3020'},result(3,0),'success')
    assert detector.observe('s','get_product_details',{'product_id':'P3020'},result(4,1),'success') is None


async def test_precise_missing_ids_and_all_skus_even_when_specific_sku_requested(confirmation_env):
    e=confirmation_env;tool=build_product_details_tool(CatalogSearchUseCase(e.products),e.bus,e.evidence)
    token=ShoppingContext.set(ShoppingContextSnapshot('details','buyer','zh-CN','CNY'))
    try:
        for product_id,sku_id,missing in [('P99999',None,'P99999'),('P3020','P3020-S99','P3020-S99')]:
            data=json.loads((await tool(product_id,sku_id)).text)
            assert data['hits']==[] and data['existence_checked'] and data['missing_identifiers']==[missing]
        p=json.loads((await tool('P3020','P3020-S2')).text)['hits'][0]
        assert p['default_sku_id']=='P3020-S2' and len(p['skus'])==2
    finally:ShoppingContext.reset(token)
