"""旧流程移除后的业务回归；使用真实工具和图，不调用外部模型。"""
import json
from dataclasses import replace
import pytest
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    from tests.test_retrieval import _settings
    settings = replace(_settings(tmp_path), llm_fallback_model='', llm_min_interval_seconds=0,
        database_url=f'sqlite+aiosqlite:///{tmp_path / "workflow.db"}')
    monkeypatch.setattr('app.composition.load_settings', lambda: settings)


async def test_reference_or_excluded_id_does_not_turn_semantic_search_into_exact_lookup():
    usecase = CatalogSearchUseCase(InMemoryProductRepository())
    result = await usecase.execute(ProductSearchSpec(
        normalized_query='找类似 P1003 的其他通勤背包，排除 P1003',
        excluded_product_ids=('P1003',), top_k=10))
    assert result['recall_strategy'] != 'exact_id_lookup'
    assert result['hits']
    assert all(hit['product_id'] != 'P1003' for hit in result['hits'])


async def test_view_replaces_visible_referents_and_supports_quote_without_candidate_permission(tmp_path, monkeypatch):
    from tests.test_agent_handoff import ScriptedModel, call
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput
    from langchain_core.messages import AIMessage, ToolMessage
    model = ScriptedModel(responses=[call('show_product_details', {
        'targets':[{'product_id':'P3020'},{'product_id':'P3021'},{'product_id':'P1003'}], 'guidance':'查看这两款。'}),
        call('quote_products', {'items':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':2}],
            'ship_to':'CN','currency':'CNY'}), AIMessage(content='报价已取得。')])
    c = await _container(tmp_path, monkeypatch, model)
    try:
        first = await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','查看颈枕的两个平台和背包。'))
        assert first.error is None and first.presentations[0]['kind'] == 'product_view'
        await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','第二款背包两件多少钱？'))
        hint = next(m for m in model.seen[-1] if m.name == 'candidate_state')
        groups = json.loads(hint.content.split('\n',1)[1])['candidates']
        assert [[item['product_id'] for item in g['items']] for g in groups] == [['P3020','P3021'],['P1003']]
        assert not await c.orchestrator._evidence_store.find_product('buyer','s',product_id='P1003',sku_id='P1003-S1')
        state = (await c.orchestrator._sessions._agents['s'].graph.aget_state(c.orchestrator._sessions._agents['s'].config)).values
        quote = next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name == 'quote_products')
        assert quote.status == 'success'
        assert quote.artifact['data']['quote']['items'][0]['quantity'] == 2
        from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
        from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
        token = ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'))
        try:
            page = await build_conversation_fact_lookup(c.orchestrator._evidence_store)(batch=0,position=1)
            assert [h['product_id'] for h in page.data['records'][0]['data']['hits']] == ['P3020','P3021']
        finally:
            ShoppingContext.reset(token)
    finally:
        await c.shutdown()


async def test_display_batches_cannot_grant_candidate_permission(tmp_path):
    from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
    store = ContextEvidenceStore(tmp_path/'evidence.db')
    hit = {'product_id':'P1003','skus':[{'sku_id':'P1003-S1'}]}
    await store.save('buyer','s','display_batch',{'hits':[hit]})
    assert await store.find_product('buyer','s',product_id='P1003',sku_id='P1003-S1') is None
    assert await store.find_observed_product('buyer','s',product_id='P1003',sku_id='P1003-S1')
    await store.save('buyer','s','comparison',{'hits':[{**hit,'constraint_issues':['over_price_cap']}]})
    assert await store.find_product('buyer','s',product_id='P1003',sku_id='P1003-S1') is None
    await store.save('buyer','s','recommendation',{'hits':[{**hit,'constraint_issues':[]}]})
    assert await store.find_product('buyer','s',product_id='P1003',sku_id='P1003-S1')
    assert await store.find_observed_product('other','s',product_id='P1003',sku_id='P1003-S1') is None


async def test_completed_trade_preparation_is_not_an_undelivered_research_task(tmp_path, monkeypatch):
    from dataclasses import asdict
    from tests.test_agent_handoff import ScriptedModel, call, submission
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput
    from app.domain.order.address import Address
    from langchain_core.messages import AIMessage
    line = {'product_id':'P1003','sku_id':'P1003-S1','quantity':1}
    address = Address('合成买家','CN','浙江','杭州','合成测试路1号','310000','13800000000')
    parent = ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),
        call('select_products',{'items':[line]}),call('task_dispatch',{
            'subagent_type':'trade_agent','task':{'goal':'按已选商品准备确认'}}),AIMessage(content='等待确认卡操作。')])
    child = ScriptedModel(responses=[call('create_order_tool',{'items':[line],'shipping_address':asdict(address)}),
        submission(status='completed')])
    c = await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.trade_agent.create_chat_model',lambda *a,**kw:child)
    try:
        before = await c.trade_store.get_inventory()
        result = await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','选一件背包，请专家准备确认。'))
        assert result.status == 'completed' and result.error is None
        state = (await c.orchestrator._sessions._agents['s'].graph.aget_state(c.orchestrator._sessions._agents['s'].config)).values
        assert not state['task_plan']['tasks']
        cards = await c.trade_store.list_confirmations(buyer_id='buyer',session_id='s')
        assert len(cards)==1 and cards[0]['status']=='pending'
        assert await c.trade_store.get_inventory()==before
        assert (await c.trade_store.list_orders(buyer_id='buyer'))['total']==0
    finally:
        await c.shutdown()


@pytest.mark.parametrize('child_status,expected', [('failed','partial'),('needs_input','needs_input')])
async def test_incomplete_trade_delegation_keeps_actual_outcome(tmp_path,monkeypatch,child_status,expected):
    from tests.test_agent_handoff import ScriptedModel,call,submission
    from tests.test_langgraph_runtime import _container
    from app.application.usecases.intents import SubmitIntentInput
    from langchain_core.messages import AIMessage
    parent = ScriptedModel(responses=[call('task_dispatch',{'subagent_type':'trade_agent',
        'task':{'goal':'核查订单准备取消'}}),AIMessage(content='已说明剩余事项。')])
    child = ScriptedModel(responses=([call('query_order_tool',{'order_id':'synthetic-missing'}),
        submission(status='failed',issues=['没有找到订单'])] if child_status=='failed' else
        [submission(status='needs_input',questions=['请提供订单号'])]))
    c = await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.trade_agent.create_chat_model',lambda *a,**kw:child)
    try:
        result = await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','请专家核查订单。'))
        assert result.status==expected
    finally:
        await c.shutdown()
