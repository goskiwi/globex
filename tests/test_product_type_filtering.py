"""商品类型是目录事实；从状态到研究及交付共用筛选，未知不由标题补造。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.application.agents.shopping_state import Filters, ShoppingWork, ShoppingUpdate, apply_update, compile_search
from app.application.tools.product_details_tool import build_product_details_tool
from app.application.tools.recommendation_tools import build_recommendation_tool, Pick
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.domain.catalog.sku import Sku
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.persistence.seed_products import _product_from_record
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from tests.trade_test_helpers import confirmation_env


def item(identifier, kind, category='旅行装备'):
    return Product(product_id=identifier, title='通勤背包', brand='测试', category=category,
        product_type=kind, origin_country='CN', description='可放在背包里的电脑配件', ships_to=['CN'],
        skus=[Sku(identifier+'-S1', '标准', Money.of(5000, 'CNY'), 3)])


@pytest.mark.parametrize('mode', ['keyword', 'vector', 'hybrid', 'hybrid_vector'])
async def test_every_recall_path_uses_explicit_type_not_title_or_category(mode):
    products = [item('P9901','背包'), item('P9902','背包','办公学习'),
                item('P9903','电脑支架'), item('P9904',None)]
    index = SimpleNamespace(search=AsyncMock(return_value=[VectorHit(p.product_id, 1) for p in products]),
        search_filtered=AsyncMock(side_effect=lambda vector, top_n, product_ids:
            [VectorHit(pid, 1) for pid in product_ids]))
    catalog = CatalogSearchUseCase(InMemoryProductRepository(products),
        hybrid_enabled=mode.startswith('hybrid'),
        embedder=SimpleNamespace(embed=AsyncMock(return_value=[1])) if 'vector' in mode else None,
        vector_index=index if 'vector' in mode else None)
    result = await catalog.execute(ProductSearchSpec(normalized_query='通勤背包', product_type='背包', top_k=10))
    assert {p['product_id'] for p in result['hits']} == {'P9901', 'P9902'}
    assert all(p['product_type']=='背包' for p in result['hits'])
    if mode=='hybrid_vector':
        assert set(index.search_filtered.await_args.kwargs['product_ids'])=={'P9901','P9902'}


@pytest.mark.parametrize('kind,reason', [('电脑支架','product_type_mismatch'), (None,'product_type_unknown')])
async def test_exact_search_explains_mismatch_and_unknown_without_hiding_details(tmp_path, kind, reason):
    product = item('P9901', kind)
    catalog = CatalogSearchUseCase(InMemoryProductRepository([product]))
    result = await catalog.execute(ProductSearchSpec(product_id='P9901', product_type='背包'))
    assert result['hits']==[] and result['existence_checked']
    assert reason in result['filtered_out'][0]['skus'][0]['reasons']
    effective = compile_search(ShoppingWork(filters=Filters(product_type='背包')))
    token = ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY', effective_search=effective))
    try:
        details = await build_product_details_tool(catalog, TradeEventBus(), ContextEvidenceStore(tmp_path/'e.db'))('P9901')
        assert details.ok and details.data['hits'][0]['product_id']=='P9901'
        assert details.data['hits'][0]['product_type']==kind
        assert reason in details.data['hits'][0]['skus'][0]['constraint_issues']
    finally:
        ShoppingContext.reset(token)


def test_type_update_clear_and_reset_do_not_change_other_conditions():
    before = ShoppingWork(filters=Filters(product_type='背包', price_max_major=300, ship_to='CN'), source_message_id='u')
    changed = apply_update(before, ShoppingUpdate(filters=Filters(product_type='耳机')), 'u')
    assert changed.filters.product_type=='耳机' and changed.filters.price_max_major==300 and changed.filters.ship_to=='CN'
    assert apply_update(changed, ShoppingUpdate(filters=Filters(product_type=None)), 'u').filters.product_type is None
    assert apply_update(changed, ShoppingUpdate(reset=True), 'u').filters.product_type is None
    with pytest.raises(ValidationError):
        Filters(product_type='backpack')
    with pytest.raises(ValueError):
        ProductSearchSpec(normalized_query='包',product_type='双肩包')


def test_catalog_type_is_required_and_null_stays_unknown():
    from scripts.generate_catalog_fixture import build_records
    record=build_records()[0]
    missing={key:value for key,value in record.items() if key!='product_type'}
    with pytest.raises(KeyError):
        _product_from_record(missing)
    unknown=_product_from_record({**record,'title':'明示背包','product_type':None})
    assert unknown.product_type is None, '运行时不从标题补类型'


async def test_recommendation_cannot_bypass_type_using_previously_read_product(confirmation_env):
    env=confirmation_env
    catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    saved=await catalog.execute(ProductSearchSpec(product_id='P1003'))
    await env.evidence.save('buyer-1','session-1','products',saved)
    effective=compile_search(ShoppingWork(filters=Filters(product_type='耳机')))
    token=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY',effective_search=effective))
    try:
        result=await build_recommendation_tool(catalog, env.evidence, env.bus)([Pick(product_id='P1003', sku_id='P1003-S1', quantity=1, reason='已读的背包')], '按当前需要选择')
        assert not result.ok and 'product_type_mismatch' in result.data
    finally:
        ShoppingContext.reset(token)


async def test_parallel_research_keeps_type_in_each_scope_and_candidate_facts(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call,submission
    from tests.test_langgraph_runtime import _container
    from tests.test_task_execution import ask
    steps=[{'id':'bag','goal':'找背包','filters':{'product_type':'背包','price_max_major':300}},
           {'id':'ear','goal':'找耳机','filters':{'product_type':'耳机','price_max_major':600}}]
    children=iter([ScriptedModel(responses=[call('product_search_tool',{'product_id':pid}),
        submission(candidates=[{'product_id':pid,'sku_id':pid+'-S1','assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}])])
        for pid in ('P1003','P1022')])
    main=ScriptedModel(responses=[*[call('set_shopping_requirements', {'changes': {'filters': {'ship_to':'CN','target_currency':'CNY'}}}), call('set_research_plan', {'plan': steps})],
        AIMessage(content='',tool_calls=[{'id':s['id'],'name':'task_dispatch','args':{
            'subagent_type':'search_agent','task':{'goal':s['goal'],'step_id':s['id']}}} for s in steps]),
        call('present_research',{'reports': [{'task_id': task_id, 'title': task_id} for task_id in ['bag', 'ear']]}, turn_action='finish')])
    container=await _container(tmp_path,monkeypatch,main)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:next(children))
    try:
        result,state=await ask(container,'请专家分别研究300元背包和600元耳机，寄中国，返回候选SKU')
        assert result.status=='completed'
        assert state['shopping_work']['filters']['product_type'] is None
        for message in state['messages']:
            if isinstance(message,ToolMessage) and message.name=='task_dispatch':
                data=message.artifact['data']
                kind=data['scope']['effective_search']['parameters']['product_type']
                assert kind==('背包' if data['scope']['task_id']=='bag' else '耳机')
                assert all(c['facts']['product']['product_type']==kind for c in data['candidates'])
    finally:
        await container.shutdown()
