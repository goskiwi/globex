"""真实 LangGraph、SQLite checkpoint 与 AG-UI 的审批恢复验证。"""
from __future__ import annotations
from app.application.memory.preference_selector import PreferenceSelector
import pytest
from ag_ui.core import RunAgentInput

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.agents.orchestrator import SubmitIntentInput
from app.application.runtime.events import RequireUserConfirm, ApprovalCall, approval_event
from app.infrastructure.ag_ui_journal import AGUIJournal, JournalConflict
from tests.test_langgraph_runtime import _container, ScriptedModel, PreferenceStore
from tests.test_ag_ui_journal import body


@pytest.mark.parametrize("approved", [True, False])
async def test_native_pause_persist_resume_and_no_replay(tmp_path, monkeypatch, approved):
    store = PreferenceStore()
    container = await _container(tmp_path, monkeypatch, ScriptedModel(mode="approval"))
    def bind(c):
        c.orchestrator._sessions._main_factory._preference_store = store
        c.orchestrator._sessions._main_factory._preference_selector = PreferenceSelector()
    bind(container)
    try:
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput("s", "buyer", "zh-CN", "CNY", "记住我喜欢轻便设计"))
        assert result.error is None and not store.items
        assert "确认或拒绝" in result.final_text
        session = container.orchestrator._sessions._agents["s"]
        snapshot = await session.graph.aget_state(session.config)
        assert len(snapshot.interrupts) == 1
        pending = approval_event(snapshot.interrupts)
        identifier = pending.reply_id + ":" + pending.tool_calls[0].id
        await container.shutdown()
        container = await _container(tmp_path, monkeypatch, ScriptedModel(mode="approval"))
        bind(container)
        foreign = await container.orchestrator.handle_intent(
            SubmitIntentInput("s", "buyer", "zh-CN", "CNY", "确认",
                confirmations=({"interrupt_id": "foreign", "approved": True},)))
        assert foreign.error and not store.items
        decision = SubmitIntentInput("s", "buyer", "zh-CN", "CNY", "确认",
            confirmations=({"interrupt_id": identifier, "approved": approved},))
        resolved = await container.orchestrator.handle_intent(decision)
        assert resolved.error is None and len(store.items) == int(approved)
        if approved:
            assert resolved.final_text == "偏好已保存"
        restored = container.orchestrator._sessions._agents["s"]
        assert not (await restored.graph.aget_state(restored.config)).interrupts
        repeated = await container.orchestrator.handle_intent(decision)
        assert repeated.error and len(store.items) == int(approved)
    finally:
        await container.shutdown()


async def test_agui_interrupt_and_journal_decision_identity(tmp_path):
    request = body()
    events = []
    adapter = AGUIRunAdapter(RunAgentInput.model_validate(request), events.append)
    adapter.start()
    adapter.on_agent_event(RequireUserConfirm("langgraph", [
        ApprovalCall("interrupt.call", "remember_preference_tool", '{"statement":"裙子"}'),
    ]))
    adapter.finish("", "completed", None)
    end = events[-1].model_dump(mode="json", by_alias=True)
    assert end["outcome"]["type"] == "interrupt"
    journal = AGUIJournal(tmp_path / "journal.db")
    await journal.reserve(request, "b1", "owner")
    await journal.append("r1", "owner", [
        event.model_dump(mode="json", by_alias=True, exclude_none=True) for event in events])
    restored = AGUIJournal(tmp_path / "journal.db")
    identifier = "langgraph:interrupt.call"
    assert (await restored.run("r1", "b1"))["state"]["toolApprovals"][0]["id"] == identifier
    resume = body("r2")
    resume["resume"] = [{"interruptId": identifier, "status": "resolved", "payload": {"approved": True}}]
    await restored.reserve(resume, "b1", "owner2")
    changed = {**resume, "resume": [
        {"interruptId": identifier, "status": "resolved", "payload": {"approved": False}}]}
    with pytest.raises(JournalConflict):
        await restored.reserve(changed, "b1", "owner3")
