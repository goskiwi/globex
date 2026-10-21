"""商品观察拥有独立对象，跨等待不会被其他操作刷新。"""
import asyncio
from types import SimpleNamespace

import pytest
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product import Product
from app.domain.catalog.sku import Sku
from app.domain.catalog.money import Money
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


def product():
    return Product('P8800','合成通勤背包','测试','旅行装备','CN','通勤背包',
        ships_to=['CN'],product_type='背包', dimensions_cm={'height':40},
        skus=[Sku('P8800-S1','标准',Money.from_major_units(50,'CNY'),5)])


@pytest.mark.parametrize('entry', ['find_by_id','find_by_ids','list_all'])
async def test_all_reads_own_nested_data_and_constructor_input(entry):
    seed=product(); repo=InMemoryProductRepository([seed]);seed.skus[0].stock=999
    seed.dimensions_cm['height']=999
    calls=[]
    async def inventory(ids):
        calls.append(ids);return {sku:5 for sku in ids}
    repo.bind_inventory(inventory)
    async def read():
        if entry=='find_by_id':return await repo.find_by_id('P8800')
        if entry=='find_by_ids':return (await repo.find_by_ids(['P8800']))[0]
        return (await repo.list_all())[0]
    first=await read();first.skus[0].stock=0;first.dimensions_cm['height']=0;first.ships_to.clear()
    second=await read()
    assert second.skus[0].stock==5 and second.dimensions_cm=={'height':40} and second.ships_to==['CN']
    assert calls==[['P8800-S1'],['P8800-S1']]
    assert repo._products['P8800'].skus[0].stock==5


@pytest.mark.parametrize('mode',['hybrid','vector'])
@pytest.mark.parametrize('pause',['embed','rank'])
async def test_search_keeps_observation_during_inventory_changes(mode,pause):
    repo=InMemoryProductRepository([product()]);stock=5
    async def inventory(ids):return {sku:stock for sku in ids}
    repo.bind_inventory(inventory)
    entered,release=asyncio.Event(),asyncio.Event()
    async def wait(stage):
        if stage==pause:
            entered.set();await release.wait()
    async def embed(text):await wait('embed');return [1.0]
    async def rank(query,documents):await wait('rank');return [1.0]*len(documents)
    async def vector(*args,**kwargs):return [VectorHit('P8800',1.0)]
    catalog=CatalogSearchUseCase(repo,embedder=SimpleNamespace(embed=embed),
        vector_index=SimpleNamespace(search=vector,search_filtered=vector),
        reranker=SimpleNamespace(rerank=rank),hybrid_enabled=mode=='hybrid')
    pending=asyncio.create_task(catalog.execute(ProductSearchSpec(normalized_query='通勤背包',product_type='背包',ship_to='CN',price_max_major=100)))
    async with asyncio.timeout(5):
        await entered.wait();stock=0
        assert (await repo.find_by_id('P8800')).skus[0].stock==0
        release.set();result=await pending
    assert result['hits'][0]['skus'][0]['stock']==5
    assert (await repo.list_all())[0].skus[0].stock==0
