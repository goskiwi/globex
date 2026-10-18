"""跨执行、传输与维护边界的故障场景。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.application.agents.context_service import ContextService
from app.application.runtime.sessions import SessionCoordinator
from app.domain.session.ports.session_store import SessionNotFound, StaleSessionWrite
from tests.test_ag_ui import collect, make_orchestrator
from tests.test_confirmation_service import prepare, approve
from tests.trade_test_helpers import confirmation_env  # noqa: F401


@pytest.mark.parametrize("evidence_failure", [False, True])
async def test_final_event_keeps_version_and_is_audited_after_required_saves(tmp_path, evidence_failure):
    from dataclasses import replace
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.application.runtime.execution import ExecutionResult
    from app.application.runtime.request_scope import ShoppingContext
    from app.application.usecases.intents import SubmitIntentInput
    from app.infrastructure.eventbus import observe_run_events
    from app.infrastructure.persistence.sql.repositories import create_engine, bootstrap_schema, SqlConversationStore
    from app.infrastructure.persistence.sql.tables import ConversationEventRow

    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
    await bootstrap_schema(engine)
    orchestrator, _, _ = make_orchestrator()
    orchestrator._conversation_store = SqlConversationStore(engine)
    if evidence_failure:
        orchestrator._evidence_store = SimpleNamespace(search=AsyncMock(return_value=[]),
            save=AsyncMock(side_effect=OSError("synthetic evidence failure")))
    async def reply(*args):
        ShoppingContext.set(replace(ShoppingContext.current(), prompt_version="synthetic-version",
                                    capability_digest="synthetic-capability"))
        return ExecutionResult("synthetic answer", "completed")
    orchestrator._reply = reply
    events = []
    try:
        with observe_run_events(events.append):
            result = await orchestrator.handle_intent(SubmitIntentInput("s", "buyer", "zh-CN", "CNY", "synthetic request"))
        final = next(e for e in events if e.type == "final.result")
        assert final.correlation["prompt_version"] == "synthetic-version"
        assert final.correlation["capability_digest"] == "synthetic-capability"
        assert result.status == ("failed" if evidence_failure else "completed")
        async with async_sessionmaker(engine)() as db:
            rows = list(await db.scalars(select(ConversationEventRow).where(ConversationEventRow.type == "final.result")))
            assert len(rows) == 1
            assert rows[0].payload["status"] == result.status
            assert rows[0].payload["text"] == result.final_text
        assert ShoppingContext.current() is None
    finally:
        await engine.dispose()


async def test_persistence_failure_reaches_agui_without_success_terminal():
    orchestrator, agent, sessions = make_orchestrator()
    sessions.persist = AsyncMock(side_effect=OSError("synthetic save failure"))
    events = await collect(orchestrator)
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "PERSISTENCE_FAILED"
    assert "核查" in events[-1]["message"]
    assert not any(e["type"] == "RUN_FINISHED" for e in events)
    assert agent.calls == 1


async def test_compaction_cannot_overlook_pending_operation_behind_history(confirmation_env):
    env = confirmation_env
    await prepare(env)
    for _ in range(20):
        await approve(env, await prepare(env), approved=False)
    coordinator = SessionCoordinator()
    store = SimpleNamespace(
        context_operation=AsyncMock(side_effect=SessionNotFound("not found")),
        create_context_operation=AsyncMock(),
    )
    service = ContextService(SimpleNamespace(session_coordinator=coordinator), store, None, env.service)
    with pytest.raises(StaleSessionWrite, match="待确认"):
        await service.start("session-1", "buyer-1", "compact", 0)
    store.create_context_operation.assert_not_awaited()
    assert coordinator.active_sessions == 0


async def test_reply_text_cannot_masquerade_as_cancel_signal():
    from app.application.runtime.execution import ExecutionResult
    from app.application.usecases.intents import SubmitIntentInput
    orchestrator, _, sessions = make_orchestrator()
    orchestrator._reply = AsyncMock(return_value=ExecutionResult("[cancelled] 这个标记的含义", "completed"))
    sessions.persist = AsyncMock(side_effect=OSError("synthetic save failure"))
    result = await orchestrator.handle_intent(SubmitIntentInput("s", "buyer", "zh-CN", "CNY", "解释标记"))
    assert result.error_code == "PERSISTENCE_FAILED"


async def test_compaction_rechecks_pending_operations_after_cross_process_lease():
    import asyncio
    from contextlib import asynccontextmanager
    leased = False
    @asynccontextmanager
    async def lease(session):
        nonlocal leased
        leased = True  # 模拟等待另一实例释放会话时，新确认单已提交。
        yield None
    confirmations = SimpleNamespace(pending=AsyncMock(side_effect=lambda *args: ["new-confirmation"] if leased else []))
    sessions = SimpleNamespace(invalidate=AsyncMock(), get_or_create=AsyncMock())
    orchestrator = SimpleNamespace(session_coordinator=SessionCoordinator(), sessions=sessions, session_lease=lease)
    operation = {"operation_id": "compact", "session_id": "s", "expected_revision": 0}
    store = SimpleNamespace(
        context_operation=AsyncMock(side_effect=SessionNotFound("missing")),
        create_context_operation=AsyncMock(return_value=(operation, True)),
        context_view=AsyncMock(return_value={"revision": 0}),
        finish_context_operation=AsyncMock(), renew_context_operation=AsyncMock(),
    )
    service = ContextService(orchestrator, store, None, confirmations)
    await service.start("s", "buyer", "compact", 0)
    await asyncio.gather(*list(service.tasks.values()))
    sessions.get_or_create.assert_not_awaited()
    assert store.finish_context_operation.await_args.args[2] == "failed"
    assert store.finish_context_operation.await_args.args[3]["error_code"] == "StaleSessionWrite"
    assert orchestrator.session_coordinator.active_sessions == 0
