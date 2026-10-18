"""缓存回收不删除checkpoint、不丢失活跃执行票据。"""
from langchain_core.messages import HumanMessage

from app.application.agents.main_agent import SessionRegistry
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from tests.test_session_fencing import LocalFactory, stores, graph_factories  # noqa: F401


async def load(registry, session_id, text=None):
    token = ShoppingContext.set(ShoppingContextSnapshot(session_id, "buyer", "zh-CN", "CNY"))
    try:
        graph = await registry.get_or_create(session_id)
        if text:
            await graph.graph.ainvoke({"messages": [HumanMessage(content=text)]}, graph.config)
        return graph
    finally:
        ShoppingContext.reset(token)


async def test_lru_eviction_reloads_persisted_history(stores, graph_factories):
    (store, _), _ = stores
    registry = SessionRegistry(graph_factories[0], store, max_cached_sessions=2)
    for session_id in ("a", "b"):
        await load(registry, session_id, "synthetic " + session_id)
        assert await registry.persist(session_id)
    await load(registry, "a")
    assert await registry.persist("a")
    await load(registry, "c", "synthetic c")
    assert await registry.persist("c")
    assert set(registry._agents) == {"a", "c"}
    assert set(registry._claims) == {"a", "c"}
    restored = await load(registry, "b")
    snapshot = await restored.graph.aget_state(restored.config)
    assert [m.content for m in snapshot.values["messages"]] == ["synthetic b"]
    assert await registry.persist("b")
    assert len(registry._agents) == len(registry._claims) == 2


async def test_cache_pressure_keeps_all_active_session_claims(stores, graph_factories):
    (store, _), _ = stores
    registry = SessionRegistry(graph_factories[0], store, max_cached_sessions=1)
    await load(registry, "a", "first")
    await load(registry, "b", "second")
    assert set(registry._claims) == {"a", "b"}
    assert await registry.persist("b")
    assert "a" in registry._claims
    assert await registry.persist("a")
    assert len(registry._agents) == len(registry._claims) == 1
    assert not registry._active
