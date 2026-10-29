"""以真实存储、应用图和商品装配覆盖四个实际反例，不只核对最终PASS。"""
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.application.agents.shopping_state import ShoppingWork, ShoppingUpdate, Filters, apply_update, compile_search
from app.application.tools.recommendation_tools import Pick, build_comparison_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.buyer.preference import BuyerPreference
from tests.test_semantic_memory import make
from tests.trade_test_helpers import confirmation_env


def test_memory_input_errors_remain_correctable_and_service_failure_is_not_success():
    from app.application.memory.outcomes import memory_completion
    from app.application.runtime.results import rejected_call
    call={'id':'m','name':'remember_preference_tool'}
    assert memory_completion([rejected_call(call,'参数缺失',code='invalid_input')]) is None
    assert memory_completion([rejected_call(call,'版本过期',code='stale_memory')]) is None
    text=memory_completion([rejected_call(call,'外部服务不可用',code='unavailable')])
    assert '未取得成功回执' in text and '已保存' not in text


async def test_memory_write_returns_actual_created_and_reused_records(tmp_path):
    store=make(tmp_path)
    request=BuyerPreference('buyer','like','喜欢裙子',source_kind='agent',source_ref='session-current')
    first=await store.append(request)
    saved=(await store.list_by_buyer('buyer'))[0]
    assert first.facts==(saved,)
    assert first.created_ids==(saved.memory_id,)
    assert saved.source_ref=='session-current' and saved.version==1
    second=await store.append(request)
    assert second.created_ids==() and second.facts==(saved,)
    assert len(await store.audit('buyer',saved.memory_id))==1


@pytest.mark.parametrize('approved',[True,False])
async def test_memory_approval_closes_from_receipt_without_state_reinterpretation(tmp_path,monkeypatch,approved):
    from tests.test_langgraph_runtime import _container, ScriptedModel, PreferenceStore
    from app.application.runtime.events import approval_event
    from app.application.usecases.intents import SubmitIntentInput
    model=ScriptedModel(mode='approval');store=PreferenceStore()
    container=await _container(tmp_path,monkeypatch,model)
    container.orchestrator._sessions._main_factory._preference_store=store
    try:
        await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','记住我喜欢轻便设计'))
        session=container.orchestrator._sessions._agents['s']
        pending=approval_event((await session.graph.aget_state(session.config)).interrupts)
        assert pending and not store.items
        count=len(model.seen)
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','页面决定',
            confirmations=({'interrupt_id':pending.reply_id+':'+pending.tool_calls[0].id,'approved':approved},)))
        assert not result.error and len(model.seen)==count
        assert len(store.items)==int(approved)
        assert result.final_text==('已保存长期偏好：喜欢轻便设计。' if approved else '本次长期记忆操作已拒绝，未执行。')
        assert not (await session.graph.aget_state(session.config)).interrupts
    finally:await container.shutdown()


def test_search_assumptions_cannot_be_written_as_buyer_preferences():
    work=ShoppingWork(filters=Filters(),source_message_id='u1',latest_request='买通勤背包，想便宜一点。')
    with pytest.raises(ValueError,match='买家原文'):
        apply_update(work,ShoppingUpdate(preferences=['通勤外观简洁']),'u1')
    changed=apply_update(work,ShoppingUpdate(preferences=['便宜一点']),'u1')
    assert changed.preference_sources=={'便宜一点':'u1'}
    changed.source_message_id='u2';changed.latest_request='预算改成200，偏好保持。'
    kept=apply_update(changed,ShoppingUpdate(preferences=['便宜一点']),'u2')
    assert kept.preference_sources=={'便宜一点':'u1'}
    cleared=apply_update(kept,ShoppingUpdate(preferences=[]),'u2')
    assert not cleared.preferences and not cleared.preference_sources


async def test_unknown_fit_is_explicit_and_quote_disclosure_comes_from_pricing(confirmation_env):
    env=confirmation_env
    catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    for pid in ('P1003','P1018'):
        await env.evidence.save('buyer-1','session-1','products',await catalog.execute(ProductSearchSpec(product_id=pid)))
    work=ShoppingWork(filters=Filters(ship_to='CN',target_currency='CNY',landed_budget_major=300),
        requirements=['能装15到16寸笔记本'])
    token=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY',effective_search=compile_search(work)))
    try:
        result=await build_comparison_tool(catalog,env.evidence,env.bus)(
            [Pick(product_id=p,sku_id=p+'-S1',quantity=1,evidence_fields=['dimensions_cm','description']) for p in ('P1003','P1018')],
            'P1018-S1',['requirements','price'])
        assert result.ok
        data=result.data
        assert data['unverified_requirements']==work.requirements
        assert '优先核实' in data['guidance'] and '以下要求尚未核实' in data['guidance']
        assert '包含本次报价中的运费与税费' in data['guidance']
        for card in data['hits']:
            assert card['requirement_checks']==[{'requirement':work.requirements[0],'status':'unverified'}]
            assert card['landed_price']['tariff_minor']==0
            assert card['decision_evidence'][0]['value']==card['dimensions_cm']
            assert '足以装下' not in card['recommendation_reason']
    finally:ShoppingContext.reset(token)


async def test_explicit_catalog_capability_is_preserved_as_source_not_discarded(confirmation_env):
    env=confirmation_env
    product=env.products._products['P1003']
    product.description='厂商标注：专用内仓支持16英寸笔记本，内仓尺寸为36×26×3厘米。'
    catalog=CatalogSearchUseCase(env.products,pricing=env.service._pricing)
    await env.evidence.save('buyer-1','session-1','products',await catalog.execute(ProductSearchSpec(product_id='P1003')))
    from app.application.tools.recommendation_tools import build_recommendation_tool
    token=ShoppingContext.set(ShoppingContextSnapshot('session-1','buyer-1','zh-CN','CNY'))
    try:
        result=await build_recommendation_tool(catalog,env.evidence,env.bus)([Pick(product_id='P1003',sku_id='P1003-S1',quantity=1)],[])
        card=result.data['hits'][0]
        assert card['decision_evidence'][0]['value']==product.description
        assert product.description in card['recommendation_reason']
        assert '尚未核实' not in card['recommendation_reason']
        assert '到手金额尚未核算' in result.data['guidance']
    finally:ShoppingContext.reset(token)
