"""交易读取与审批并行：真实SQLite一致快照，审批不被读取写锁阻塞。"""
import asyncio
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_confirmation_service import prepare, approve
from tests.trade_test_helpers import confirmation_env  # noqa: F401


@pytest.mark.parametrize("cancel", [False, True])
async def test_approval_during_state_read_keeps_one_consistent_snapshot(confirmation_env, monkeypatch, cancel):
    env = confirmation_env
    prepared = await prepare(env)
    order_id = None
    if cancel:
        created = await approve(env, prepared)
        order_id = created["order"]["order_id"]
        prepared = await env.service.handle_order('buyer-1', 'session-1', order_id, {'reason': 'synthetic cancellation'})
    reading, release = asyncio.Event(), asyncio.Event()
    original = AsyncSession.scalars

    async def paused(db, statement, *args, **kwargs):
        result = await original(db, statement, *args, **kwargs)
        if "trade_confirmations" in str(statement) and not reading.is_set():
            reading.set()
            await release.wait()
        return result

    monkeypatch.setattr(AsyncSession, "scalars", paused)
    reader = asyncio.create_task(env.service.agent_state("buyer-1", "session-1"))
    try:
        await asyncio.wait_for(reading.wait(), 2)
        committed = await asyncio.wait_for(approve(env, prepared), 2)
    finally:
        release.set()
    state = await reader
    assert len(state["pending_confirmations"]) == 1
    if cancel:
        assert state["orders"][0]["order_id"] == order_id
        assert state["orders"][0]["status"] == "CONFIRMED"
        assert committed["order"]["status"] == "CANCELLED"
    else:
        assert state["orders"] == []
        assert committed["order"]["status"] == "CONFIRMED"
    after = await env.service.agent_state("buyer-1", "session-1")
    assert after["pending_confirmations"] == []
    assert after["orders"][0]["status"] == committed["order"]["status"]


async def test_current_transactions_survive_more_than_twenty_history_records(confirmation_env):
    env = confirmation_env
    created = await approve(env, await prepare(env))
    env.now[0] += timedelta(seconds=1)
    pending = (await prepare(env))["confirmation"]
    for _ in range(20):
        env.now[0] += timedelta(seconds=1)
        await approve(env, await prepare(env), approved=False)
    state = await env.service.agent_state("buyer-1", "session-1")
    assert state["orders"][0]["order_id"] == created["order"]["order_id"]
    assert len(state["pending_confirmations"]) == 1
    restored = (await env.service.restore("buyer-1", "session-1"))["confirmations"]
    assert len(restored) == 21
    assert pending["confirmation_id"] in {c["confirmation_id"] for c in restored}
    assert (await env.service.restore("other", "session-1"))["confirmations"] == []
    assert (await env.service.agent_state("buyer-1", "other")) == {"orders": [], "pending_confirmations": []}
