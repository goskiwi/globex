"""完整商品+SKU身份贯穿查看、比较和确认准备。"""
import pytest
from langchain_core.messages import ToolMessage,AIMessage
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container
from tests.test_workflow_boundaries import ask
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.tools.product_view_tool import build_product_view_tool
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.eventbus import TradeEventBus


@pytest.mark.parametrize('targets',[['P3004-S2','P3004-S1'],['P3004-S1','P3004-S2']])
async def test_view_multiple_skus_keeps_one_platform_record_and_order(tmp_path,targets):
    context=ShoppingContextSnapshot('s','buyer','zh-CN','CNY')
    token=ShoppingContext.set(context)
    try:
        tool=build_product_view_tool(CatalogSearchUseCase(InMemoryProductRepository()),ContextEvidenceStore(tmp_path/'e.db'),TradeEventBus())
        result=await tool([{'product_id':'P3004','sku_id':sid} for sid in targets],'查看同款两个规格')
        assert result.ok and len(result.data['hits'])==1
        card=result.data['hits'][0]
        assert card['requested_sku_ids']==targets and card['default_sku_id']==targets[0]
        assert len(card['skus'])==2 and 'requested_sku_id' not in card
        assert ShoppingContext.current()==context and not context.selected_lines
        invalid=await tool([{'product_id':'P3004','sku_id':targets[0]},{'product_id':'P3004','sku_id':'P3004-S999'}],'查看指定规格')
        assert not invalid.ok
    finally:ShoppingContext.reset(token)


async def test_sibling_comparison_then_prepare_second_sku_without_searching_again(tmp_path,monkeypatch):
    address={'recipient_name':'合成测试买家','country':'CN','state':'浙江','city':'杭州','address_line':'合成测试路1号','postal_code':'310000','phone':'13800000000'}
    parent=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1001'}),call('compare_products',{'entries':[{'product_id':'P1001','sku_id':sid,'quantity':1,'reason':'比较颜色规格'} for sid in ['P1001-S1','P1001-S2']],'preferred_sku_id':'P1001-S1','dimensions':['颜色'],'guidance':'按颜色偏好选择。'}, finish=True),call('select_products',{'items': [{'product_id':'P1001','sku_id':'P1001-S2','quantity':2}]}),call('create_order_tool',{'items':[{'product_id': sku.split('-S')[0], 'sku_id': sku, 'quantity': 2} for sku in ['P1001-S2']],'shipping_address':address}, finish=True),AIMessage(content='确认单已准备，请核对。')])
    c=await _container(tmp_path,monkeypatch,parent)
    try:
        compared,_=await ask(c,'比较同款两个规格，之后再选择。')
        assert compared.status=='completed' and compared.presentations[0]['kind']=='comparison'
        inventory=await c.trade_store.get_inventory(['P1001-S2'])
        result,state=await ask(c,'选第二个规格两件，按已给地址准备确认，不代我批准。')
        receipt=next(m for m in reversed(state['messages']) if isinstance(m,ToolMessage) and m.name=='create_order_tool')
        assert receipt.status=='success',receipt.text
        data=receipt.artifact['data']
        assert data['confirmation_required'] and data['confirmation']['status']=='pending'
        line=data['confirmation']['payload']['items'][0]
        assert line['product_id']=='P1001' and line['sku_id']=='P1001-S2' and line['quantity']==2
        assert parent.cursor==4 and await c.trade_store.get_inventory(['P1001-S2'])==inventory
        assert (await c.trade_store.list_orders(buyer_id='buyer'))['total']==0
    finally:await c.shutdown()
