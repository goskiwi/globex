"""临时真实SQLite+生产式库存绑定，测同次推荐/创建准备的重复观察读取。"""
import json
from pathlib import Path
from app.application.agents.shopping_state import ShoppingWork,Filters,compile_search
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.tools.recommendation_tools import build_recommendation_tool,Pick
from app.application.usecases.order_usecases import OrderItemInput
from tests.trade_test_helpers import confirmation_env,test_address as address

async def test_operation_inventory_reads(confirmation_env):
    e=confirmation_env;reads=[]
    original=e.store.get_inventory
    async def observed(sku_ids=None):
        reads.append({'sku_count':len(sku_ids) if sku_ids is not None else None})
        return await original(sku_ids)
    e.store.get_inventory=observed
    e.products.bind_inventory(observed)
    catalog=CatalogSearchUseCase(e.products,pricing=e.service._pricing)
    policy=compile_search(ShoppingWork(filters=Filters(price_max_major=200,target_currency='CNY',ship_to='CN')))
    token=ShoppingContext.set(ShoppingContextSnapshot('read-scope','audit-buyer','zh-CN','CNY',effective_search=policy))
    try:
        result=await catalog.execute(ProductSearchSpec(product_id='P1003',target_currency='CNY',ship_to='CN'))
        await e.evidence.save('audit-buyer','read-scope','products',result)
        reads.clear()
        returned=await build_recommendation_tool(catalog, e.evidence, e.bus)([Pick(product_id='P1003', sku_id='P1003-S1', quantity=1)], [])
        assert returned.ok
        recommendation=list(reads)
        assert len(recommendation)==1
        reads.clear()
        pending=await e.service.prepare_order('audit-buyer','read-scope',
            [OrderItemInput('P1003','P1003-S1',1)],address(),'CNY')
        preparation=list(reads)
        assert len(preparation)==1 and pending['confirmation']['status']=='pending'
        assert (await e.store.list_orders(buyer_id='audit-buyer'))['total']==0
    finally:ShoppingContext.reset(token)

async def test_batch_details_quote_and_duplicate_sku_quantities_share_one_observation(confirmation_env):
    from app.application.tools.product_view_tool import build_product_view_tool,ProductTarget
    from app.application.usecases.pricing import QuoteItem
    e=confirmation_env;reads=[]
    async def inventory(ids):
        reads.append(ids);return await e.store.get_inventory(ids)
    e.products.bind_inventory(inventory)
    catalog=CatalogSearchUseCase(e.products,pricing=e.service._pricing)
    token=ShoppingContext.set(ShoppingContextSnapshot('batch','buyer','zh-CN','CNY'))
    try:
        response=await build_product_view_tool(catalog,e.evidence,e.bus)(
            [ProductTarget(product_id='P1003'),ProductTarget(product_id='P2101')],'查看两款全部规格。')
        assert response.ok and len(reads)==1
        reads.clear()
        result=await e.service._pricing.quote([QuoteItem('P1003','P1003-S1',1),QuoteItem('P1003','P1003-S1',2),
            QuoteItem('P1001','P1001-S1',1)],'CN','CNY')
        assert len(reads)==1 and len(result['items'])==2
        assert next(line['quantity'] for line in result['items'] if line['sku_id']=='P1003-S1')==3
    finally:ShoppingContext.reset(token)
