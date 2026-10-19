"""跨币种规格的搜索、详情和研究共用读取时显示口径。"""
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository

async def test_search_and_details_share_current_sku_display():
    catalog=CatalogSearchUseCase(InMemoryProductRepository())
    spec=ProductSearchSpec(product_id='P40018',target_currency='CNY',ship_to='CN')
    search=(await catalog.execute(spec))['hits'][0]
    identifier=search['default_sku_id']
    details=(await catalog.product_details('P40018',spec,sku_id=identifier))['hits'][0]
    first=next(s for s in search['skus'] if s['sku_id']==identifier)
    second=next(s for s in details['skus'] if s['sku_id']==identifier)
    for key in ('price_major','currency','display_price_major','display_currency'):
        assert first[key]==second[key]
    assert first['currency']=='SGD' and first['price_major']==54.94
    assert first['display_currency']=='CNY' and first['display_price_major']==291.18
    assert search['landed_price']['total_amount_minor']==31618
