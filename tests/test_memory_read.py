"""主动读取不混淆全量、相关片段和查询失败；真实SQLite与主图验证归属和后续修改。"""
import json
import pytest
from langchain_core.messages import ToolMessage

from app.application.tools.read_preferences_tool import build_read_preferences_tool,build_search_preferences_tool
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.memory.preference_resolution import resolve_preferences
from app.application.usecases.intents import SubmitIntentInput
from app.application.runtime.events import approval_event
from app.domain.buyer.preference import BuyerPreference
from app.infrastructure.semantic_memory import SemanticPreferenceStore,MemoryUnavailable
from tests.test_semantic_memory import Extract
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container


class Embed:
    fail=False
    calls=0
    async def embed(self,text):
        self.calls+=1
        if self.fail:raise RuntimeError('synthetic embedding unavailable')
        if '鞋' in text:return [-1.,0.,0.]
        if '裙' in text or '衣' in text:return [1.,0.,0.]
        if '塑料' in text:return [0.,1.,0.]
        return [0.,0.,1.]
    async def embed_batch(self,texts):return [await self.embed(t) for t in texts]


async def populated(tmp_path,count=12):
    store=SemanticPreferenceStore(tmp_path/'memory.db',Extract(),Embed(),'test',.5)
    for i in range(count):await store.append(BuyerPreference('buyer','like',f'喜欢裙子款式{i}'))
    await store.append(BuyerPreference('other','like','喜欢其他买家的私人款式'))
    return store


@pytest.fixture
def identity():
    token=ShoppingContext.set(ShoppingContextSnapshot('session','buyer','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)


async def test_complete_listing_exceeds_auto_hint_without_embedding(tmp_path,identity):
    store=await populated(tmp_path)
    resolved=await resolve_preferences(store,store,'buyer','裙子',5)
    assert len(resolved.selected)==5 and len(resolved.facts)==12
    assert '非完整列表' in resolved.hint and '12条' in resolved.hint
    before=store.embedder.calls;store.embedder.fail=True
    result=await build_read_preferences_tool(store)()
    assert result.ok and store.embedder.calls==before
    assert result.data['scope']=='all' and result.data['total_count']==result.data['returned_count']==12
    expected=await store.list_by_buyer('buyer')
    assert {p['memory_id'] for p in result.data['preferences']}=={p.memory_id for p in expected}
    assert all(p['version']==1 and p['source_kind']=='user' for p in result.data['preferences'])
    assert all(p.statement in result.data['guidance'] for p in expected)
    assert '其他买家' not in result.text


async def test_related_search_limits_matches_but_not_shopping_constraints(tmp_path,identity):
    store=await populated(tmp_path)
    await store.append(BuyerPreference('buyer','dislike','不要塑料'))
    tool=build_search_preferences_tool(store)
    result=await tool(query='裙子',limit=3)
    assert result.ok
    assert (result.data['total_count'],result.data['candidate_count'],result.data['returned_count'])==(13,12,3)
    assert result.data['has_more'] and all(p['kind']=='like' for p in result.data['candidates'])
    selected=await store.select(await store.list_by_buyer('buyer'),'裙子',3)
    assert len(selected)==4 and selected[0].statement=='不要塑料'
    negative=await build_read_preferences_tool(store)(kind='dislike')
    assert negative.data['total_count']==1 and negative.data['preferences'][0]['statement']=='不要塑料'


async def test_no_match_empty_store_and_service_failure_have_distinct_results(tmp_path,identity):
    store=await populated(tmp_path,1);tool=build_search_preferences_tool(store)
    none=await tool(query='鞋')
    assert none.ok and none.data['total_count']==1 and none.data['candidate_count']==0
    empty=await build_read_preferences_tool(store)(kind='dislike')
    assert empty.ok and empty.data['total_count']==0 and '没有已保存的禁忌' in empty.data['guidance']
    store.embedder.fail=True
    failure=await tool(query='裙子')
    assert not failure.ok and failure.data['status']=='unavailable'
    assert '不能据此判断没有保存' in failure.data['guidance']
    assert 'total_count' not in failure.data
    assert len(await store.select(await store.list_by_buyer('buyer'),'裙子',5))==0
    assert (await build_read_preferences_tool(store)()).data['returned_count']==1


async def test_failed_vector_rebuild_is_not_reported_as_no_match(tmp_path,identity):
    store=await populated(tmp_path,1)
    with store._db() as db:db.execute("UPDATE memory_facts SET model_id='previous-model'")
    async def unavailable(texts):raise RuntimeError('synthetic rebuild unavailable')
    store.embedder.embed_batch=unavailable
    result=await build_search_preferences_tool(store)(query='裙子')
    assert not result.ok and result.data['status']=='unavailable'
    assert '不能据此判断没有保存' in result.data['guidance']
    assert (await build_read_preferences_tool(store)()).data['returned_count']==1


@pytest.mark.parametrize('change',['delete','update'])
async def test_query_never_returns_stale_record_after_async_change(tmp_path,change):
    store=await populated(tmp_path,1);original=(await store.list_by_buyer('buyer'))[0]
    embed=store.embedder.embed
    async def mutate(text):
        store.embedder.embed=embed
        if change=='delete':await store.delete_by_id('buyer',original.memory_id,original.version)
        else:await store.replace_by_id('buyer',original.memory_id,original.version,BuyerPreference('buyer','like','喜欢新款裙子'))
        return await embed(text)
    store.embedder.embed=mutate
    result=await store.search('buyer','裙子')
    if change=='delete':assert result.total_count==0 and not result.matches
    else:
        assert result.matches[0].preference.version==2
        assert result.matches[0].preference.statement=='喜欢新款裙子'


async def test_native_complete_read_delivers_all_200_without_model_summary(tmp_path,monkeypatch):
    store=await populated(tmp_path,200)
    model=ScriptedModel(responses=[call('read_preferences_tool',{},turn_action='finish')])
    container=await _container(tmp_path,monkeypatch,model)
    factory=container.orchestrator._sessions._main_factory
    factory._preference_store=factory._preference_selector=store
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','列出我的全部记忆'))
        assert not result.error and model.cursor==1
        assert result.final_text.count('\n')==200 and '完整列表' in result.final_text
        assert '喜欢裙子款式199' in result.final_text and '其他买家' not in result.final_text
        session=container.orchestrator._sessions._agents['s']
        state=await session.graph.aget_state(session.config)
        receipt=next(m for m in state.values['messages'] if isinstance(m,ToolMessage) and m.name=='read_preferences_tool')
        assert receipt.artifact['data']['returned_count']==200
        assert not state.interrupts
    finally:await container.shutdown()


async def test_read_then_update_uses_real_identity_and_still_requires_approval(tmp_path,monkeypatch):
    store=await populated(tmp_path,1);saved=(await store.list_by_buyer('buyer'))[0]
    model=ScriptedModel(responses=[call('search_preferences_tool',{'query':'裙子'}),
        call('update_preference_tool',{'memory_id':saved.memory_id,'expected_version':saved.version,
            'previous_statement':saved.statement,'kind':'like','statement':'喜欢连衣裙'},turn_action='finish')])
    container=await _container(tmp_path,monkeypatch,model)
    factory=container.orchestrator._sessions._main_factory
    factory._preference_store=factory._preference_selector=store
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','查出裙子偏好，改成喜欢连衣裙'))
        assert not result.error and (await store.list_by_buyer('buyer'))[0]==saved
        receipt=next(m for m in model.seen[-1] if isinstance(m,ToolMessage) and m.name=='search_preferences_tool')
        assert json.loads(receipt.content)['candidates'][0]['memory_id']==saved.memory_id
        session=container.orchestrator._sessions._agents['s']
        pending=approval_event((await session.graph.aget_state(session.config)).interrupts)
        assert pending is not None
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','批准',
            confirmations=({'interrupt_id':pending.reply_id+':'+pending.tool_calls[0].id,'approved':True},)))
        actual=(await store.list_by_buyer('buyer'))[0]
        assert not result.error and actual.memory_id==saved.memory_id and actual.version==2
        assert actual.statement=='喜欢连衣裙' and model.cursor==2
    finally:await container.shutdown()


async def test_reader_has_no_identity_override_and_no_missing_identity_access(tmp_path):
    from app.application.runtime.tools import as_langchain_tool
    store=await populated(tmp_path,1)
    fn=build_read_preferences_tool(store)
    tool=as_langchain_tool(fn,read_only=True,turn_delivery=True)
    assert 'buyer_id' not in tool.tool_call_schema.model_fields
    assert 'query' not in tool.tool_call_schema.model_fields
    assert tool.tool_call_schema.model_config['extra']=='forbid'
    token=ShoppingContext.set(None)
    try:assert not (await fn()).ok
    finally:ShoppingContext.reset(token)


async def test_search_candidates_are_not_automatically_shown_as_semantic_matches(tmp_path,monkeypatch):
    store=await populated(tmp_path,12)
    chosen=min(await store.list_by_buyer('buyer'),key=lambda p:p.memory_id)
    model=ScriptedModel(responses=[call('search_preferences_tool',{'query':'裙子'}),
        call('read_preferences_tool',{'memory_ids':[chosen.memory_id]},turn_action='finish')])
    container=await _container(tmp_path,monkeypatch,model)
    factory=container.orchestrator._sessions._main_factory
    factory._preference_store=factory._preference_selector=store
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','请查找相关偏好后展示一条。'))
        assert not result.error and model.cursor==2
        search=next(m for m in model.seen[-1] if isinstance(m,ToolMessage) and m.name=='search_preferences_tool')
        assert len(json.loads(search.content)['candidates'])==5
        assert result.final_text.count('\n')==1 and chosen.statement in result.final_text
        assert '选中的1条' in result.final_text and '完整列表' not in result.final_text
    finally:await container.shutdown()


async def test_selected_read_cannot_access_other_buyers_ids(tmp_path,identity):
    store=await populated(tmp_path,1)
    foreign=(await store.list_by_buyer('other'))[0]
    result=await build_read_preferences_tool(store)(memory_ids=[foreign.memory_id])
    assert not result.ok and '其他买家' not in result.text


@pytest.mark.parametrize('invalid',[{'memory_ids':42},{'buyer_id':'other'}])
async def test_invalid_read_call_can_correct_without_leaking_or_breaking_process(tmp_path,monkeypatch,invalid):
    store=await populated(tmp_path,1)
    model=ScriptedModel(responses=[call('read_preferences_tool',invalid,turn_action='finish'),
        call('read_preferences_tool',{},turn_action='finish')])
    container=await _container(tmp_path,monkeypatch,model)
    factory=container.orchestrator._sessions._main_factory
    factory._preference_store=factory._preference_selector=store
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY','查看我的记忆'))
        assert not result.error and model.cursor==2 and '其他买家' not in result.final_text
        assert '共保存1条' in result.final_text
        session=container.orchestrator._sessions._agents['s']
        state=await session.graph.aget_state(session.config)
        first=next(m for m in state.values['messages'] if isinstance(m,ToolMessage) and m.name=='read_preferences_tool')
        assert first.status=='error' and first.artifact['error_code']=='invalid_input'
    finally:await container.shutdown()
