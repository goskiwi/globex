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
    from app.application.runtime.execution import ExecutionResult
    from app.application.runtime.request_scope import ShoppingContext
    from app.infrastructure.eventbus import observe_run_events
    from app.application.runtime.runs import RunService
    from app.infrastructure.ag_ui_journal import AGUIJournal
    from ag_ui.core import RunAgentInput
    orchestrator, _, _ = make_orchestrator()
    journal=AGUIJournal(tmp_path/'journal.db')
    runtime=RunService(journal,orchestrator)
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
        body=RunAgentInput.model_validate({'threadId':'s','runId':'run','state':{},'tools':[],'context':[],
            'messages':[{'id':'u','role':'user','content':'synthetic request'}],'forwardedProps':{'buyerId':'buyer'}})
        with observe_run_events(events.append):
            await runtime.start(body,'buyer')
            run=await runtime.wait('run','buyer',10)
        result=run['result']
        final = next(e for e in events if e.type == "final.result")
        assert final.correlation["prompt_version"] == "synthetic-version"
        assert final.correlation["capability_digest"] == "synthetic-capability"
        assert result["status"] == ("failed" if evidence_failure else "completed")
        frames,_,_=await journal.events('run','buyer')
        assert frames[-1]['event']['type']==('RUN_ERROR' if evidence_failure else 'RUN_FINISHED')
        if evidence_failure:
            assert run['state']['executionError']==result['text'] and run['status']=='error'
        else:
            assert run['state']['executionStatus']==result['status']
            assert run['messages'][-1]['content']==result['text']
        assert any(e['event'].get('name')=='usage.summary' for e in frames)
        assert ShoppingContext.current() is None
    finally:
        await runtime.shutdown()


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


async def test_shutdown_finishes_compaction_cancelled_before_its_first_schedule(tmp_path, monkeypatch):
    import asyncio
    from app.application.usecases.intents import SubmitIntentInput
    from tests.test_langgraph_runtime import _container, ScriptedModel
    container = await _container(tmp_path, monkeypatch, ScriptedModel())
    try:
        await container.orchestrator.handle_intent(SubmitIntentInput("s", "buyer", "zh-CN", "CNY", "synthetic request"))
        service = container.context_service
        revision = (await service.store.context_view("s", "buyer"))["revision"]
        operation = await service.start("s", "buyer", "before-start", revision)
        service.tasks[operation["operation_id"]].cancel()
        await service.shutdown()
        saved = await service.store.context_operation(operation["operation_id"], "buyer")
        assert saved["status"] == "interrupted"
        assert not service.tasks and not service._cleanup_tasks
        assert container.orchestrator.session_coordinator.active_sessions == 0
        # 不必等待30秒租约，新的整理可立即创建并完成。
        following = await service.start("s", "buyer", "following", revision)
        await asyncio.gather(*list(service.tasks.values()))
        assert (await service.store.context_operation(following["operation_id"], "buyer"))["status"] == "noop"
    finally:
        await container.shutdown()


async def test_unsaved_result_cannot_be_published_as_success():
    orchestrator, agent, sessions = make_orchestrator()
    sessions.persist = AsyncMock(return_value=False)
    events = await collect(orchestrator)
    assert agent.calls == 1
    assert events[-1]['type'] == 'RUN_ERROR' and events[-1]['code'] == 'PERSISTENCE_FAILED'
    assert '核查' in events[-1]['message']
    assert not any(e['type'] == 'RUN_FINISHED' for e in events)
