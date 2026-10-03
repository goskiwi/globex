"""跨审批、重启与并发撤回：真实主图和 SQLite 验证执行约束，不调用外部模型。"""
import json
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from app.application.agents.orchestrator import SubmitIntentInput
from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
from app.application.memory.preference_resolution import resolve_preferences
from app.application.runtime.events import approval_event
from app.domain.buyer.preference import BuyerPreference
from app.infrastructure.semantic_memory import SemanticPreferenceStore, MemoryUnavailable
from tests.test_langgraph_runtime import _container
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_semantic_memory import Embed


class Extract:
    fail = False
    async def extract(self, p, existing=()):
        if self.fail:
            raise MemoryUnavailable('测试模拟提炼失败')
        return [{'kind': p.kind, 'statement': p.statement, 'evidence': p.statement,
                 'constraint': {'material_tags': ['合成聚合物'], 'category': None}
                 if p.kind == 'dislike' else None}]


def bind(container, store):
    factory = container.orchestrator._sessions._main_factory
    factory._preference_store = factory._preference_selector = store


async def ask(container, text, **kwargs):
    result = await container.orchestrator.handle_intent(
        SubmitIntentInput('s', 'buyer', 'zh-CN', 'CNY', text, **kwargs))
    assert result.error is None, result
    return result


async def state(container):
    session = container.orchestrator._sessions._agents['s']
    return await session.graph.aget_state(session.config)


# 独立业务分支逐项覆盖，重启只选批准/拒绝代表场景，避免无效笛卡尔积。
@pytest.mark.parametrize('operation,approved,restart', [
    ('remember', True, False), ('forget', True, False), ('update', True, False),
    ('unrelated', True, False), ('failed', True, False),
    ('remember', False, False), ('forget', False, False), ('update', False, False),
    ('unrelated', True, True), ('forget', False, True),
])
async def test_memory_decision_refreshes_search_and_hint(tmp_path, monkeypatch, operation, approved, restart):
    store = SemanticPreferenceStore(tmp_path/'memory.db', Extract(), Embed(), 'test', .5)
    if operation != 'remember':
        await store.append(BuyerPreference('buyer', 'dislike', '不要合成聚合物'))
        saved = (await store.list_by_buyer('buyer'))[0]
    if operation in {'remember', 'unrelated', 'failed'}:
        name = 'remember_preference_tool'
        args = {'kind': 'dislike' if operation == 'remember' else 'like',
                'statement': '不要合成聚合物' if operation == 'remember' else '喜欢轻便设计'}
    else:
        name = operation + '_preference_tool'
        args = {'memory_id': saved.memory_id, 'expected_version': saved.version}
        if operation == 'forget':
            args['statement'] = saved.statement
        else:
            args.update(previous_statement=saved.statement, kind='like', statement='喜欢轻便设计')
    model = ScriptedModel(responses=[
        call('product_search_tool', {'product_id': 'P1001'}), call(name, args),
        call('product_search_tool', {'product_id': 'P1001'}), AIMessage(content='核验完毕'),
        call('product_search_tool', {'product_id': 'P1001', 'sku_id': 'P1001-S1'}), AIMessage(content='核验完毕')])
    container = await _container(tmp_path, monkeypatch, model)
    bind(container, store)
    try:
        await ask(container, '先核验商品，再按我的要求修改长期偏好')
        pending = approval_event((await state(container)).interrupts)
        assert pending is not None
        if restart:
            await container.shutdown()
            container = await _container(tmp_path, monkeypatch, model)
            bind(container, store)
        if operation == 'failed':
            store.distiller.fail = True
        await ask(container, '确认' if approved else '拒绝', confirmations=({
            'interrupt_id': pending.reply_id + ':' + pending.tool_calls[0].id,
            'approved': approved},))
        current = await state(container)
        assert not current.interrupts
        forbidden = (approved if operation == 'remember' else
                     not approved if operation in {'forget', 'update'} else True)
        hint = [m for m in model.seen[-1] if m.name == 'memory_hint']
        assert len(hint) == 1
        assert ('不要合成聚合物' in hint[0].text) == forbidden
        await ask(container, '再次核验商品')
        messages = (await state(container)).values['messages']
        searches = [json.loads(m.content) for m in messages
                    if isinstance(m, ToolMessage) and m.name == 'product_search_tool']
        assert all('hits' in m for m in searches), searches
        assert [len(m['hits']) for m in searches] == [1 if operation == 'remember' else 0,
                                                     0 if forbidden else 1, 0 if forbidden else 1]
        assert searches[-1]['query_conditions']['excluded_material_tags'] == (
            ['合成聚合物'] if forbidden else [])
    finally:
        await container.shutdown()


@pytest.mark.parametrize('change', ['delete', 'replace', 'add'])
async def test_async_recall_and_executable_facts_use_same_read(tmp_path, change):
    embed = Embed()
    store = SemanticPreferenceStore(tmp_path/'memory.db', Extract(), embed, 'test', .5)
    await store.append(BuyerPreference('buyer', 'dislike', '不要合成聚合物'))
    original = (await store.list_by_buyer('buyer'))[0]
    await store.append(BuyerPreference('buyer', 'like', '喜欢轻便设计'))
    original_embed = embed.embed
    async def change_while_ranking(query):
        embed.embed = original_embed
        if change == 'delete':
            await store.delete_by_id('buyer', original.memory_id, original.version)
        elif change == 'replace':
            await store.replace_by_id('buyer', original.memory_id, original.version,
                                     BuyerPreference('buyer', 'like', '喜欢帆布'))
        else:
            await store.append(BuyerPreference('buyer', 'dislike', '不购买含合成聚合物的东西'))
        return await original_embed(query)
    embed.embed = change_while_ranking
    resolved = await resolve_preferences(store, store, 'buyer', '选购背包', 5)
    assert list(resolved.facts) == await store.list_by_buyer('buyer')
    effective = compile_search(ShoppingWork(filters=Filters()), resolved.facts)
    assert bool(effective['parameters']['excluded_material_tags']) == (change == 'add')
    assert ('不要合成聚合物' in resolved.hint) == (change == 'add')
    if change == 'add':
        assert '不购买含合成聚合物的东西' in resolved.hint


async def test_memory_write_and_business_batch_neither_executes(tmp_path, monkeypatch):
    store = SemanticPreferenceStore(tmp_path/'memory.db', Extract(), Embed(), 'test', .5)
    model = ScriptedModel(responses=[AIMessage(content='', tool_calls=[
        {'name': 'remember_preference_tool', 'args': {'kind': 'like', 'statement': '喜欢轻便设计'}, 'id': 'm'},
        {'name': 'product_search_tool', 'args': {'product_id': 'P1001'}, 'id': 's'}]),
        AIMessage(content='请分批执行')])
    container = await _container(tmp_path, monkeypatch, model)
    bind(container, store)
    try:
        await ask(container, '记住偏好并搜索')
        current = await state(container)
        assert not current.interrupts and not await store.list_by_buyer('buyer')
        results = [m for m in current.values['messages'] if isinstance(m, ToolMessage)]
        assert len(results) == 2
        assert all(m.status == 'error' and '本批均未执行' in m.content for m in results)
    finally:
        await container.shutdown()


async def test_resumed_preferences_reach_child_search_and_final_recommendation(tmp_path, monkeypatch):
    from tests.test_agent_handoff import submission
    store = SemanticPreferenceStore(tmp_path/'memory.db', Extract(), Embed(), 'test', .5)
    child = ScriptedModel(responses=[call('product_search_tool', {'product_id': 'P1001'}),
        submission(status='partial', summary='没有符合条件的商品', unmet_constraints=['材质不符合'])])
    model = ScriptedModel(responses=[
        call('product_search_tool', {'product_id': 'P1001'}),
        call('remember_preference_tool', {'kind': 'dislike', 'statement': '不要合成聚合物'}),
        call('task_dispatch', {'subagent_type': 'search_agent', 'task': {'goal': '核验P1001'}}),
        call('recommend_products', {'picks': [{'product_id': 'P1001', 'sku_id': 'P1001-S1',
             'quantity': 1, 'reason': '旅行收纳'}], 'mode': 'alternatives',
             'preferred_sku_id': 'P1001-S1', 'dimensions': [], 'guidance': '旅行收纳选择'}),
        AIMessage(content='该商品不符合材质要求')])
    container = await _container(tmp_path, monkeypatch, model)
    bind(container, store)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model', lambda *a, **kw: child)
    try:
        await ask(container, '记住不要合成聚合物，然后让子任务核验商品')
        event = approval_event((await state(container)).interrupts)
        await ask(container, '确认', confirmations=({'interrupt_id': event.reply_id + ':' + event.tool_calls[0].id,
                                                  'approved': True},))
        child_search = next(m for m in child.seen[-1] if isinstance(m, ToolMessage)
                            and m.name == 'product_search_tool')
        data = json.loads(child_search.content)
        assert not data['hits']
        assert data['query_conditions']['excluded_material_tags'] == ['合成聚合物']
        result = next(m for m in (await state(container)).values['messages']
                      if isinstance(m, ToolMessage) and m.name == 'recommend_products')
        assert result.status == 'error'
        assert 'material_excluded' in result.content or '材质' in result.content
    finally:
        await container.shutdown()
