from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
"""共同交付语义、容量、首选与业务事实，不将自然语言选择变成规则评分。"""
import json
import pytest
from pydantic import ValidationError
from app.application.tools.recommendation_tools import (
    RecommendationInput, ComparisonInput, Pick, MAX_DECISION_ITEMS,
    build_recommendation_tool, build_comparison_tool, build_bundle_tool,
)
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env


def picks():
    return [Pick(product_id=pid, sku_id=f'{pid}-S1', quantity=1)
            for pid in ('P1003', 'P1018', 'P1049', 'P3004')]


def test_recommendation_and_comparison_own_their_delivery_fields():
    entries=picks()
    recommendation=RecommendationInput(picks=entries, focus=[])
    assert set(RecommendationInput.model_fields)=={'picks','focus'}
    for model,field,values in [
        (RecommendationInput,'picks',{'picks': entries, 'focus': []}),
        (ComparisonInput,'entries',{'entries': entries, 'preferred_sku_id': 'P1018-S1', 'dimensions': ['spec']})]:
        assert model.model_json_schema()['properties'][field]['maxItems']==MAX_DECISION_ITEMS
        with pytest.raises(ValidationError):model.model_validate({**values,field:entries*4})
        for required in model.model_json_schema()['required']:
            with pytest.raises(ValidationError):model.model_validate({k:v for k,v in values.items() if k!=required})


def test_model_cannot_submit_unsupported_capability_or_price_prose():
    raw={'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'evidence_fields':['description','dimensions_cm']}
    assert Pick.model_validate(raw).evidence_fields==raw['evidence_fields']
    for field,value in [('reason','一定容纳16寸笔记本'),('tradeoffs',['未含税'])]:
        with pytest.raises(ValidationError):Pick.model_validate({**raw,field:value})
    with pytest.raises(ValidationError):RecommendationInput(picks=[Pick.model_validate(raw)],guidance='未含税')


async def test_four_choices_share_facts_preference_focus_and_capacity(confirmation_env):
    env = confirmation_env
    catalog = CatalogSearchUseCase(env.products, pricing=env.service._pricing)
    for pick in picks():
        await env.evidence.save('buyer-1', 'session-1', 'products',
            await catalog.execute(ProductSearchSpec(product_id=pick.product_id)))
    token = ShoppingContext.set(ShoppingContextSnapshot('session-1', 'buyer-1', 'zh-CN', 'CNY',
        effective_search=compile_search(ShoppingWork(filters=Filters(**{'ship_to': 'CN', 'target_currency': 'CNY', 'landed_budget_major': 300})), currency='CNY')))
    try:
        recommend = build_recommendation_tool(catalog, env.evidence, env.bus)
        compare = build_comparison_tool(catalog, env.evidence, env.bus)
        selected = 'P1018-S1'
        focus = ['description', 'weight_kg']
        guidance = '这次更看重防雨，首选防水款；其他候选分别适合轻装或预算优先。'
        recommendation = json.loads((await recommend(picks(), focus)).text)
        comparison = json.loads((await compare(picks(), selected, focus)).text)
        assert '包含本次报价中的运费与税费' in recommendation['guidance'] and '包含本次报价中的运费与税费' in comparison['guidance']
        assert {p['default_sku_id']:p for p in recommendation['hits']} == {p['default_sku_id']:p for p in comparison['hits']}
        assert [p['default_sku_id'] for p in recommendation['hits']] == [p.sku_id for p in picks()]
        assert not {'preferred_sku_id','dimensions','max_items'} & set(recommendation)
        assert comparison['preferred_sku_id']==selected and comparison['dimensions']==['目录描述','重量（kg）']
        totals={p['default_sku_id']:p['landed_price']['total_amount_minor'] for p in comparison['hits']}
        assert totals['P1003-S1']==15400 and totals['P1018-S1']==24400 and totals['P1049-S1']==6400
        assert not (await compare(picks(),'P9999-S1',focus)).ok
        assert not (await build_bundle_tool(catalog,env.evidence,env.bus)(picks(),focus,'CN','CNY',None)).ok
        assert ShoppingContext.current().selected_lines == ()
        assert await env.store.list_confirmations(buyer_id='buyer-1',session_id='session-1') == []
    finally:
        ShoppingContext.reset(token)


async def test_catalog_evidence_is_not_rewritten_or_filtered_by_identifier(confirmation_env):
    env=confirmation_env
    catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    product=env.products._products['P1003']
    product.description='P1003-S1 的原始目录说明，未声明笔记本适配。'
    await env.evidence.save('buyer-1','session-1','products',await catalog.execute(ProductSearchSpec(product_id='P1003')))
    token=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY'))
    try:
        result=await build_recommendation_tool(catalog,env.evidence,env.bus)([Pick(product_id='P1003',sku_id='P1003-S1',quantity=1)],[])
        assert result.ok
        assert result.data['hits'][0]['decision_evidence'][0]['value']==product.description
        assert product.description in result.data['hits'][0]['recommendation_reason']
        assert await env.evidence.search('buyer-1','session-1',kind='recommendation')
    finally:ShoppingContext.reset(token)


async def test_native_agent_delivers_valid_id_explanation_without_another_generation(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput
    guidance='按你指定的P1012展示这款充气颈枕。'
    model=ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1012'}), call('recommend_products', {'picks': [{'product_id': 'P1012', 'sku_id': 'P1012-S1', 'quantity': 1}], 'focus': []}, turn_action='finish')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        result=await c.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','推荐充气颈枕'))
        assert result.status=='completed' and '候选规格' in result.final_text and model.cursor==2
        assert len(await c.orchestrator._evidence_store.search('b','s',kind='recommendation'))==1
    finally:await c.shutdown()
