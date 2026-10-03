"""物理参数来自单一演示设定，未知不补造；原始价格与销售规格独立保存。"""
import json
from pathlib import Path
import pytest
from app.domain.catalog.product import ProductHighlight
from app.domain.catalog.money import Money
from app.domain.catalog.sku import Sku
from app.infrastructure.persistence.seed_products import build_seed_products
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from scripts.catalog_seed_data import demo_product, build_demo_seed_products
from scripts.generate_catalog_fixture import DemoSpec, _legacy_record, build_records


def test_weight_wording_and_fields_share_one_value():
    product=demo_product(product_id='P9001',title='测试物品',brand='test',category='旅行装备',origin_country='CN',
        description='净重 {weight_g}g', highlights=[ProductHighlight('重量','仅 {weight_g}g')],
        unit_weight_kg=.38,skus=[Sku('P9001-S1','标准',Money.of(1000,'CNY'),1)])
    assert product.description=='净重 380g' and product.highlights[0].detail=='仅 380g'
    for index in (0,2,19,55):
        assert _legacy_record(product,index)['weight_kg']==.38
    spec=DemoSpec('{weight_g}g 三节折叠',.28)
    assert spec.label()=='280g 三节折叠' and spec.weight_kg==.28


def test_explicit_seed_weights_and_sold_unit_scope():
    products={p.product_id:p for p in build_demo_seed_products()}
    assert products['P1003'].weight_kg==.38
    assert products['P1017'].weight_kg==.12
    assert products['P1010'].weight_kg==.52  # 两支，每支260g，不把单支重当销售单位总重。
    assert products['P1047'].weight_kg==.38  # 一对190g单支。
    assert products['P1018'].weight_kg==.95
    assert products['P1052'].weight_kg==.09  # 量程50kg不是电子秤自重。
    assert products['P1003'].dimensions_cm=={'length':32,'width':24,'height':50}


async def test_unknown_weight_is_not_a_zero_weight_or_false_size_in_tool_data():
    unknown=demo_product(product_id='P9001',title='未知参数物品',brand='test',category='旅行装备',origin_country='CN',
        description='没有物理参数',skus=[Sku('P9001-S1','标准',Money.of(1000,'CNY'),1)])
    repo=InMemoryProductRepository([*build_seed_products(),unknown])
    usecase=CatalogSearchUseCase(repo)
    response=await usecase.execute(ProductSearchSpec(product_id='P9001'))
    card=response['hits'][0]
    assert 'weight_kg' not in card
    assert card['dimensions_cm']=={} and card['package_dimensions_cm']=={}
    response=await usecase.execute(ProductSearchSpec(product_id='P1003'))
    assert response['hits'][0]['weight_kg']==.38
    assert '重量：仅 380g 超轻' in response['hits'][0]['highlights']


def test_package_size_is_not_product_size_and_current_catalog_is_reproducible():
    from scripts.generate_catalog_multilingual import build_records as build_multilingual
    actual=[json.loads(line) for line in (Path(__file__).resolve().parents[1]/'data/catalog-v3.jsonl').read_text().splitlines()]
    assert actual==build_multilingual()
    packaged=next(r for r in actual if r.get('source_language'))
    assert packaged['dimensions_cm'] and packaged['package_dimensions_cm']
    assert packaged['dimensions_cm']!=packaged['package_dimensions_cm']
    products={p.product_id:p for p in build_seed_products()}
    assert products[packaged['product_id']].package_dimensions_cm==packaged['package_dimensions_cm']
    poles={r['model_spec']:r['weight_kg'] for r in build_records() if r['product_id'] in ('P3208','P3212','P3216','P3220')}
    assert poles=={'450g 不可折叠':.45,'350g 三节伸缩':.35,'280g 三节折叠':.28,'220g 五节折叠快锁':.22}


def test_loader_does_not_convert_old_missing_fields():
    from app.infrastructure.persistence.seed_products import _product_from_record
    record=build_records()[0]
    del record['package_dimensions_cm']
    with pytest.raises(KeyError): _product_from_record(record)
