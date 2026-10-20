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
    return [Pick(product_id=pid, sku_id=f'{pid}-S1', quantity=1,
                 reason='本次用途依据在商品描述中', tradeoffs=['电脑隔层资料未提供'])
            for pid in ('P1003', 'P1018', 'P1049', 'P3004')]


def test_recommendation_and_comparison_own_their_delivery_fields():
    entries=picks()
    recommendation=RecommendationInput(picks=entries,guidance='依据用途选择')
    assert set(RecommendationInput.model_fields)=={'picks','guidance'}
    for model,field,values in [
        (RecommendationInput,'picks',{'picks':entries,'guidance':'依据用途选择'}),
        (ComparisonInput,'entries',{'entries':entries,'guidance':'比较防水与背负','preferred_sku_id':'P1018-S1','dimensions':['防水','背负']})]:
        assert model.model_json_schema()['properties'][field]['maxItems']==MAX_DECISION_ITEMS
        with pytest.raises(ValidationError):model.model_validate({**values,field:entries*4})
        for required in model.model_json_schema()['required']:
            with pytest.raises(ValidationError):model.model_validate({k:v for k,v in values.items() if k!=required})


def test_reason_is_short_choice_advice_without_truncating_long_input():
    raw = {'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'选'*240,'tradeoffs':[]}
    assert Pick.model_validate(raw).reason == raw['reason']
    too_long = {**raw, 'reason':'选'*241}
    with pytest.raises(ValidationError): Pick.model_validate(too_long)
    assert len(too_long['reason']) == 241
    assert Pick.model_json_schema()['properties']['reason']['maxLength'] == 240


async def test_four_choices_share_facts_preference_focus_and_capacity(confirmation_env):
    env = confirmation_env
    catalog = CatalogSearchUseCase(env.products, pricing=env.service._pricing)
    for pick in picks():
        await env.evidence.save('buyer-1', 'session-1', 'products',
            await catalog.execute(ProductSearchSpec(product_id=pick.product_id)))
    token = ShoppingContext.set(ShoppingContextSnapshot('session-1', 'buyer-1', 'zh-CN', 'CNY',
        effective_search={'parameters':{'ship_to':'CN','target_currency':'CNY','landed_budget_major':300}}))
    try:
        recommend = build_recommendation_tool(catalog, env.evidence, env.bus)
        compare = build_comparison_tool(catalog, env.evidence, env.bus)
        selected = 'P1018-S1'
        focus = ['雨天骑行', '防水与背负']
        guidance = '这次更看重防雨，首选防水款；其他候选分别适合轻装或预算优先。'
        recommendation = json.loads((await recommend(picks(), guidance)).text)
        comparison = json.loads((await compare(picks(), selected, focus, guidance)).text)
        assert recommendation['guidance']==comparison['guidance']
        assert {p['default_sku_id']:p for p in recommendation['hits']} == {p['default_sku_id']:p for p in comparison['hits']}
        assert [p['default_sku_id'] for p in recommendation['hits']] == [p.sku_id for p in picks()]
        assert not {'preferred_sku_id','dimensions','max_items'} & set(recommendation)
        assert comparison['preferred_sku_id']==selected and comparison['dimensions']==focus
        totals={p['default_sku_id']:p['landed_price']['total_amount_minor'] for p in comparison['hits']}
        assert totals['P1003-S1']==15400 and totals['P1018-S1']==24400 and totals['P1049-S1']==6400
        assert not (await compare(picks(),'P9999-S1',focus,guidance)).ok
        assert not (await build_bundle_tool(catalog,env.evidence,env.bus)(picks(),guidance)).ok
        assert ShoppingContext.current().selected_lines == ()
        assert await env.store.list_confirmations(buyer_id='buyer-1',session_id='session-1') == []
    finally:
        ShoppingContext.reset(token)


@pytest.mark.parametrize('field',['guidance','reason','tradeoffs','dimensions'])
async def test_valid_identifiers_in_explanation_do_not_reject_catalog_choices(confirmation_env,field):
    env=confirmation_env
    catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    await env.evidence.save('buyer-1','session-1','products',
        await catalog.execute(ProductSearchSpec(product_id='P1003')))
    context=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY'))
    try:
        pick=Pick(product_id='P1003',sku_id='P1003-S1',quantity=1,reason='适合轻装出行',tradeoffs=[])
        guidance='更推荐 Wanderlite 折叠旅行双肩包，方便收纳。'
        dimensions=['收纳']
        if field=='guidance':guidance='首选 P1003，收起来方便。'
        elif field=='reason':pick=pick.model_copy(update={'reason':'推荐 P1003-S1'})
        elif field=='tradeoffs':pick=pick.model_copy(update={'tradeoffs':['P1003-S1 与其他选项不同']})
        else:dimensions=['P1003 的收纳']
        tool=build_recommendation_tool(catalog,env.evidence,env.bus)
        result=(await build_comparison_tool(catalog,env.evidence,env.bus)([pick,pick.model_copy(update={'sku_id':'P1003-S2'})], 'P1003-S1', dimensions, guidance)) if field=='dimensions' else await tool([pick],guidance)
        assert result.ok
        data=json.loads(result.text)
        assert data['guidance']==guidance and data['hits'][0]['default_sku_id']=='P1003-S1'
        assert await env.evidence.search('buyer-1','session-1',kind='comparison' if field=='dimensions' else 'recommendation')
    finally:ShoppingContext.reset(context)


async def test_native_agent_delivers_valid_id_explanation_without_another_generation(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput
    guidance='按你指定的P1012展示这款充气颈枕。'
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1012'}),call('recommend_products', {'picks': [{'product_id': 'P1012', 'sku_id': 'P1012-S1', 'quantity': 1, 'reason': '方便收纳'}], 'guidance': guidance}, finish=True)])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        result=await c.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','推荐充气颈枕'))
        assert result.status=='completed' and result.final_text==guidance and model.cursor==2
        assert len(await c.orchestrator._evidence_store.search('b','s',kind='recommendation'))==1
    finally:await c.shutdown()
