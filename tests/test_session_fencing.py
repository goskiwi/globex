"""独立连接/Registry 的持久 fencing 验证，不依赖 Redis 和外部模型。"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from contextlib import AsyncExitStack
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.application.agents.main_agent import SessionRegistry, GraphSession
from app.infrastructure.persistence.graph_checkpointer import FencedSqliteSaver
from app.domain.session.ports.session_store import SessionOwnerMismatch, SessionOwnerUnbound, StaleSessionWrite
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.persistence.json_file_stores import JsonFileSessionStore
from app.infrastructure.persistence.sql.repositories import SqlSessionStore, create_engine
from app.infrastructure.persistence.sql.tables import ConversationSessionRow


@pytest.fixture
async def stores(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    engines = [create_engine(url), create_engine(url)]
    result = [SqlSessionStore(engine) for engine in engines]
    yield result, engines
    for engine in engines:
        await engine.dispose()


async def test_new_claim_invalidates_old_writer_before_new_owner_has_saved(stores):
    (old, new), _ = stores
    first = await old.claim("session", buyer_id="buyer")
    second = await new.claim("session", buyer_id="buyer")
    assert second.fence > first.fence and second.revision == first.revision == 0
    with pytest.raises(StaleSessionWrite):
        await old.save_claim(first, '{"old":true}')
    assert await new.load("session") is None
    await new.save_claim(second, '{"new":true}')
    assert await old.load("session") == '{"new":true}'


async def test_revision_rejects_repeated_save_with_same_ticket(stores):
    (store, other), _ = stores
    claim = await store.claim("session", buyer_id="buyer")
    saved = await store.save_claim(claim, '{"v":1}')
    assert saved.revision == claim.revision + 1
    with pytest.raises(StaleSessionWrite):
        await other.save_claim(claim, '{"v":2}')
    with pytest.raises(StaleSessionWrite):
        await other.save("session", '{"bypass":true}')
    assert await other.load("session") == '{"v":1}'


async def test_delayed_save_started_before_takeover_cannot_overwrite_new_state(stores):
    (old, new), _ = stores
    first = await old.claim("session", buyer_id="buyer")
    entered, release = asyncio.Event(), asyncio.Event()
    actual = old.save_claim

    async def delayed(claim, state):
        entered.set()
        await release.wait()
        return await actual(claim, state)

    old.save_claim = delayed
    pending = asyncio.create_task(old.save_claim(first, '{"old":true}'))
    await entered.wait()
    second = await new.claim("session", buyer_id="buyer")
    await new.save_claim(second, '{"winner":true}')
    release.set()
    with pytest.raises(StaleSessionWrite):
        await pending
    assert await old.load("session") == '{"winner":true}'


async def test_independent_claims_allocate_unique_monotonic_epochs(stores):
    pair, _ = stores
    claims = await asyncio.gather(*(pair[index % 2].claim("session", buyer_id="buyer") for index in range(8)))
    assert sorted(claim.fence for claim in claims) == list(range(1, 9))
    latest = max(claims, key=lambda claim: claim.fence)
    await pair[0].save_claim(latest, '{"winner":8}')
    for stale in claims:
        if stale != latest:
            with pytest.raises(StaleSessionWrite):
                await pair[1].save_claim(stale, '{}')


async def test_save_failure_rolls_back_revision_and_snapshot(stores):
    (store, other), engines = stores
    claim = await store.claim("session", buyer_id="buyer")

    def fail(_connection, _cursor, statement, _params, _context, _many):
        if statement.startswith("INSERT INTO agent_session_states"):
            raise RuntimeError("保存快照失败")

    event.listen(engines[0].sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError):
            await store.save_claim(claim, '{"v":1}')
    finally:
        event.remove(engines[0].sync_engine, "before_cursor_execute", fail)
    assert await other.load("session") is None
    saved = await other.save_claim(claim, '{"retry":true}')
    assert saved.revision == 1


async def test_owner_binding_is_persistent_and_can_be_explicitly_disabled(stores):
    (store, other), _ = stores
    await store.assert_owner("session", "buyer-a", create=True)
    await other.assert_owner("session", "buyer-a")
    with pytest.raises(SessionOwnerMismatch):
        await other.claim("session", buyer_id="buyer-b")
    await other.claim("session", buyer_id="buyer-b", enforce_owner=False)
    with pytest.raises(SessionOwnerMismatch):
        await other.assert_owner("session", "buyer-b")


async def test_legacy_sql_owner_is_inferred_only_from_existing_history(stores):
    (store, other), engines = stores
    await store.save("legacy", '{"v":1}')
    async with async_sessionmaker(engines[0])() as db:
        db.add(ConversationSessionRow(session_id="legacy", buyer_id="original-buyer"))
        await db.commit()
    with pytest.raises(SessionOwnerMismatch):
        await other.claim("legacy", buyer_id="intruder")
    restored = await other.claim("legacy", buyer_id="original-buyer")
    assert restored.state_json == '{"v":1}'


async def test_unowned_legacy_snapshot_requires_explicit_migration(stores):
    (store, _), _ = stores
    await store.save("legacy", '{"v":1}')
    with pytest.raises(SessionOwnerUnbound):
        await store.claim("legacy", buyer_id="buyer")
    await store.bind_legacy_owner("legacy", "buyer")
    assert (await store.claim("legacy", buyer_id="buyer")).state_json == '{"v":1}'


async def test_file_mode_migrates_once_and_rejects_cross_instance_stale_writer(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "conversations").mkdir()
    legacy = tmp_path / "sessions" / "session.json"
    legacy.write_text('{"legacy":true}', encoding="utf-8")
    (tmp_path / "conversations" / "session.jsonl").write_text('{"buyer_id":"buyer"}\n', encoding="utf-8")
    one, two = JsonFileSessionStore(tmp_path), JsonFileSessionStore(tmp_path)
    try:
        first = await one.claim("session", buyer_id="buyer")
        assert first.state_json == '{"legacy":true}'
        second = await two.claim("session", buyer_id="buyer")
        await two.save_claim(second, '{"fresh":true}')
        with pytest.raises(StaleSessionWrite):
            await one.save_claim(first, '{"stale":true}')
        legacy.write_text('{"changed_file":true}', encoding="utf-8")
        assert await JsonFileSessionStore(tmp_path).load("session") == '{"fresh":true}'
        with pytest.raises(SessionOwnerMismatch):
            await one.assert_owner("session", "another")
    finally:
        await one.close()
        await two.close()


async def test_file_mode_does_not_guess_owner_or_allow_filename_aliasing(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "old.json").write_text('{}', encoding="utf-8")
    store = JsonFileSessionStore(tmp_path)
    try:
        with pytest.raises(SessionOwnerUnbound):
            await store.claim("old", buyer_id="buyer")
        with pytest.raises(SessionOwnerUnbound):
            await store.claim("o/ld", buyer_id="buyer")
    finally:
        await store.close()


class LocalFactory:
    capability_registry = None
    skill_catalog_mode = "append_only"

    def __init__(self, saver=None):
        self.saver = saver or InMemorySaver()

    def build(self):
        graph = StateGraph(MessagesState)
        graph.add_node("finish", lambda state: {})
        graph.add_edge(START, "finish")
        graph.add_edge("finish", END)
        return GraphSession(graph.compile(checkpointer=self.saver), {}, frozenset())


@pytest.fixture
async def graph_factories(stores, tmp_path):
    pair, _ = stores
    async with AsyncExitStack() as stack:
        factories = []
        for store in pair:
            saver = await stack.enter_async_context(FencedSqliteSaver.from_conn_string(str(tmp_path / "graph.db")))
            saver.session_store = store
            factories.append(LocalFactory(saver))
        yield factories


async def registry_agent(registry, content=None):
    token = ShoppingContext.set(ShoppingContextSnapshot("session", "buyer", "zh-CN", "CNY"))
    try:
        session = await registry.get_or_create("session")
        if content:
            await session.graph.ainvoke({"messages": [HumanMessage(content=content)]}, session.config)
        return session
    finally:
        ShoppingContext.reset(token)


async def test_two_registries_reject_delayed_persist_and_reload_latest_revision(stores, graph_factories):
    (old_store, new_store), _ = stores
    old, new = SessionRegistry(graph_factories[0], old_store), SessionRegistry(graph_factories[1], new_store)
    old_agent = await registry_agent(old)
    entered, release = asyncio.Event(), asyncio.Event()
    original = old_store.save_claim

    async def delayed(claim, state):
        entered.set()
        await release.wait()
        return await original(claim, state)

    old_store.save_claim = delayed
    pending = asyncio.create_task(old.persist("session"))
    await entered.wait()
    new_agent = await registry_agent(new, "新的已确认上下文")
    assert await new.persist("session") is True
    release.set()
    with pytest.raises(StaleSessionWrite):
        await pending
    assert "session" not in old._agents
    restored = await registry_agent(old)
    state = await restored.graph.aget_state(restored.config)
    assert [item.content for item in state.values["messages"]] == ["新的已确认上下文"]


async def test_claim_fence_is_inherited_without_losing_context_constraints(stores):
    (store, _), _ = stores
    registry = SessionRegistry(LocalFactory(), store)
    token = ShoppingContext.set(ShoppingContextSnapshot("session", "buyer", "zh-CN", "CNY", ("plastic",)))
    try:
        await registry.get_or_create("session")
        assert ShoppingContext.current().session_fence == 1
        assert ShoppingContext.current().preference_facts == ("plastic",)
        from app.application.runtime.preferences import PreferenceStateMiddleware
        from types import SimpleNamespace
        async def handler(request):
            assert ShoppingContext.current().session_fence == 1
            assert ShoppingContext.current().preference_facts == ()
        await PreferenceStateMiddleware(None, None, 5).awrap_tool_call(
            SimpleNamespace(state={"preference_snapshot": {"buyer_id": "buyer", "facts": []}}), handler)
        assert ShoppingContext.current().session_fence == 1
        assert ShoppingContext.current().preference_facts == ("plastic",)
    finally:
        ShoppingContext.reset(token)
    assert ShoppingContext.current() is None


async def test_file_preferences_do_not_alias_distinct_authenticated_buyers(tmp_path):
    from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore
    from app.domain.buyer.preference import BuyerPreference
    store = JsonFilePreferenceStore(tmp_path)
    await store.append(BuyerPreference("a/b", "dislike", "隐私偏好"))
    assert await store.list_by_buyer("ab") == []
    await store.append(BuyerPreference("ab", "like", "另一个人的偏好"))
    assert [item.statement for item in await store.list_by_buyer("a/b")] == ["隐私偏好"]
