"""共同交付语义、容量、首选与业务事实，不将自然语言选择变成规则评分。"""
import json
import pytest
from pydantic import ValidationError
from app.application.tools.recommendation_tools import (
    RecommendationInput, ComparisonInput, Pick, MAX_DECISION_ITEMS,
    build_recommendation_tool, build_comparison_tool,
)
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from tests.trade_test_helpers import confirmation_env


def picks():
    return [Pick(product_id=pid, sku_id=f'{pid}-S1', quantity=1,
                 reason='本次用途依据在商品描述中', tradeoffs=['电脑隔层资料未提供'])
            for pid in ('P1003', 'P1018', 'P1049', 'P3004')]


def test_shared_capacity_and_explicit_decision_fields():
    entries = picks()
    for model, field in ((RecommendationInput, 'picks'), (ComparisonInput, 'entries')):
        values = {field: entries, 'preferred_sku_id': 'P1018-S1', 'dimensions': ['雨天骑行', '背负'], 'guidance':'雨天骑车更推荐防水款；更在意轻便时可看折叠款。'}
        if field == 'picks': values['mode'] = 'alternatives'
        assert model.model_validate(values).dimensions == ['雨天骑行', '背负']
        assert model.model_json_schema()['properties'][field]['maxItems'] == MAX_DECISION_ITEMS
        for missing in ('preferred_sku_id', 'dimensions', 'guidance'):
            with pytest.raises(ValidationError):
                model.model_validate({k:v for k,v in values.items() if k != missing})
        with pytest.raises(ValidationError):
            model.model_validate({**values, field:entries * 4})


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
        recommendation = json.loads((await recommend(picks(), 'alternatives', selected, focus, guidance)).text)
        comparison = json.loads((await compare(picks(), selected, focus, guidance)).text)
        for key in ('guidance', 'hits', 'preferred_sku_id', 'dimensions', 'max_items'):
            assert recommendation[key] == comparison[key]
        assert len(recommendation['hits']) == 4
        assert recommendation['hits'][0]['default_sku_id'] == selected
        totals = {p['default_sku_id']:p['landed_price']['total_amount_minor'] for p in comparison['hits']}
        assert totals['P1003-S1'] == 15400 and totals['P1018-S1'] == 24400 and totals['P1049-S1'] == 6400
        assert not (await recommend(picks(), 'alternatives', 'P9999-S1', focus, guidance)).ok
        assert not (await recommend(picks(), 'bundle', selected, focus, guidance)).ok
        assert ShoppingContext.current().selected_lines == ()
        assert await env.store.list_confirmations(buyer_id='buyer-1',session_id='session-1') == []
    finally:
        ShoppingContext.reset(token)


@pytest.mark.parametrize('field',['guidance','reason','tradeoffs','dimensions'])
async def test_internal_ids_in_buyer_copy_are_rejected_before_publish(confirmation_env,field):
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
        bad=await tool([pick],'alternatives','P1003-S1',dimensions,guidance)
        assert not bad.ok
        assert 'Wanderlite' in bad.text and '商品名称' in bad.text
        assert not await env.evidence.search('buyer-1','session-1',kind='recommendation')
        good=await tool([Pick(product_id='P1003',sku_id='P1003-S1',quantity=1,
            reason='适合轻装出行',tradeoffs=[])],'alternatives','P1003-S1',['收纳'],
            '更推荐 Wanderlite 折叠旅行双肩包，方便收纳。')
        assert good.ok
        assert json.loads(good.text)['guidance']=='更推荐 Wanderlite 折叠旅行双肩包，方便收纳。'
    finally:ShoppingContext.reset(context)


async def test_native_agent_corrects_id_reference_before_delivery(tmp_path,monkeypatch):
    from tests.test_agent_handoff import ScriptedModel,call
    from tests.test_langgraph_runtime import _container
    from app.application.agents.orchestrator import SubmitIntentInput
    from langchain_core.messages import ToolMessage
    choice={'product_id':'P1012','sku_id':'P1012-S1','quantity':1,
            'reason':'收起来体积小，适合随身带上飞机','tradeoffs':[]}
    arguments={'picks':[choice],'mode':'alternatives','preferred_sku_id':'P1012-S1',
               'dimensions':['收纳体积'],'guidance':'我把 P1012 放在首位，收纳不占地方。'}
    guidance='更推荐 NestRest 按压式充气颈枕，收纳后只有掌心大小，不占随身行李空间。'
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1012'}),
        call('recommend_products',arguments,'bad-advice'),
        call('recommend_products',{**arguments,'guidance':guidance},'corrected-advice')])
    container=await _container(tmp_path,monkeypatch,model)
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','推荐充气颈枕'))
        assert result.status=='completed' and result.product_delivery_complete
        assert result.final_text==guidance
        feedback=[m for m in model.seen[-1] if isinstance(m,ToolMessage)
                  and m.tool_call_id.startswith('bad-advice-round-')]
        assert len(feedback)==1 and feedback[0].status=='error'
        assert 'NestRest' in feedback[0].content
        deliveries=await container.orchestrator._evidence_store.search('b','s',kind='recommendation')
        assert len(deliveries)==1
    finally:await container.shutdown()
