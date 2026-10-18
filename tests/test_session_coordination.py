"""会话锁的等待、取消、跨会话并行与生命周期回归。"""
import asyncio

import pytest

from app.application.runtime.sessions import SessionBusy, SessionCoordinator


async def test_waiters_share_lock_and_last_release_reclaims_entry():
    coordinator = SessionCoordinator()
    first = await coordinator.acquire("s")
    waiting = asyncio.create_task(coordinator.acquire("s"))
    await asyncio.sleep(0)
    assert not waiting.done()
    with pytest.raises(SessionBusy):
        await coordinator.acquire("s", wait=False)
    other = await coordinator.acquire("other", wait=False)
    other.release()
    first.release()
    second = await waiting
    assert coordinator.active_sessions == 1
    with pytest.raises(SessionBusy):
        await coordinator.acquire("s", wait=False)
    second.release()
    second.release()
    assert coordinator.active_sessions == 0


async def test_cancelled_waiter_cannot_remove_held_lock():
    coordinator = SessionCoordinator()
    first = await coordinator.acquire("s")
    waiting = asyncio.create_task(coordinator.acquire("s"))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    with pytest.raises(SessionBusy):
        await coordinator.acquire("s", wait=False)
    first.release()
    assert coordinator.active_sessions == 0


async def test_finished_sessions_do_not_accumulate_locks():
    coordinator = SessionCoordinator()
    for index in range(1000):
        async with coordinator.hold(str(index)):
            assert coordinator.active_sessions == 1
    assert coordinator.active_sessions == 0
