# -*- coding: utf-8 -*-
"""LangGraph 主图、checkpoint 与 interrupt 的最小运行回归。"""
from __future__ import annotations
from app.application.memory.preference_selector import PreferenceSelector

from tests.ag_ui_runtime_helpers import run_frames
from dataclasses import replace
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.application.agents.orchestrator import SubmitIntentInput


class ScriptedModel(BaseChatModel):
    mode: str = "plain"
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "globex-langgraph-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if self.mode == "search" and not any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="", tool_calls=[{
                "id": "search-1", "name": "product_search_tool",
                "args": {"product_id": "P1003", "sku_id": "P1003-S1"},
            }])
        elif self.mode in {"approval", "batch"} and not any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="", tool_calls=[{
                "id": "remember-1",
                "name": "remember_preference_tool",
                "args": {"kind": "like", "statement": "喜欢轻便设计"},
            }])
            if self.mode == "batch":
                message.tool_calls.append({
                    "id": "remember-2", "name": "remember_preference_tool",
                    "args": {"kind": "like", "statement": "喜欢蓝色"},
                })
        else:
            message = AIMessage(content="偏好已保存" if self.mode == "approval" else "LangGraph 运行正常")
        return ChatResult(generations=[ChatGeneration(message=message)])


class PreferenceStore:
    semantic_memory = False

    def __init__(self):
        self.items = []

    async def list_by_buyer(self, buyer_id):
        return list(self.items)

    async def append(self, preference):
        self.items.append(preference)
        return [preference]


async def _container(tmp_path, monkeypatch, model):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.composition import build_container
    from app.infrastructure.persistence.sql.repositories import bootstrap_schema
    import app.application.agents.main_agent as main_module

    container = await build_container()
    await bootstrap_schema(container.db_engine)
    await container.trade_store.initialize_inventory(await container.product_repo.list_all())
    factory = container.orchestrator._sessions._main_factory
    factory._settings = replace(container.settings, llm_fallback_model="")
    monkeypatch.setattr(main_module, "create_chat_model", lambda *args, **kwargs: model)
    return container


async def test_langgraph_roundtrip_persists_thread_state(tmp_path, monkeypatch):
    container = await _container(tmp_path, monkeypatch, ScriptedModel(mode="plain"))
    try:
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput("session", "buyer", "zh-CN", "CNY", "你好"),
            
        )
        session = container.orchestrator._sessions._agents["session"]
        state = await session.graph.aget_state(session.config)
        assert result.final_text == "LangGraph 运行正常"
        assert len(state.values["messages"]) == 3
        assert not state.interrupts
    finally:
        await container.shutdown()


async def test_batch_approval_survives_restart_and_rejects_replay(tmp_path, monkeypatch):
    from app.application.runtime.events import approval_event
    store = PreferenceStore()
    container = await _container(tmp_path, monkeypatch, ScriptedModel(mode="batch"))
    def bind(c):
        c.orchestrator._sessions._main_factory._preference_store = store
        c.orchestrator._sessions._main_factory._preference_selector = PreferenceSelector()
    bind(container)
    try:
        first = await container.orchestrator.handle_intent(
            SubmitIntentInput("batch", "buyer", "zh-CN", "CNY", "记住两个偏好"),
            
        )
        assert first.error is None and store.items == []
        session = container.orchestrator._sessions._agents["batch"]
        pending = (await session.graph.aget_state(session.config)).interrupts
        event = approval_event(pending)
        first_id = event.reply_id + ":" + event.tool_calls[0].id
        second = await container.orchestrator.handle_intent(
            SubmitIntentInput("batch", "buyer", "zh-CN", "CNY", "批准第一项",
                confirmations=({"interrupt_id": first_id, "approved": True},)),
            
        )
        assert second.error is None and store.items == []
        pending = (await session.graph.aget_state(session.config)).interrupts
        event = approval_event(pending)
        second_id = event.reply_id + ":" + event.tool_calls[0].id
        assert first_id != second_id
        await container.shutdown()
        container = await _container(tmp_path, monkeypatch, ScriptedModel(mode="batch"))
        bind(container)
        # 重启后的旧决议不能被当成本次新决议。
        stale = await container.orchestrator.handle_intent(
            SubmitIntentInput("batch", "buyer", "zh-CN", "CNY", "重复",
                confirmations=({"interrupt_id": first_id, "approved": True},)),
            
        )
        assert stale.error and not store.items
        final = await container.orchestrator.handle_intent(
            SubmitIntentInput("batch", "buyer", "zh-CN", "CNY", "拒绝第二项",
                confirmations=({"interrupt_id": second_id, "approved": False},)),
            
        )
        assert final.error is None
        assert [p.statement for p in store.items] == ["喜欢轻便设计"]
        repeated = await container.orchestrator.handle_intent(
            SubmitIntentInput("batch", "buyer", "zh-CN", "CNY", "重复",
                confirmations=({"interrupt_id": second_id, "approved": False},)),
            
        )
        assert repeated.error and len(store.items) == 1
    finally:
        await container.shutdown()


async def test_stale_execution_cannot_write_graph_checkpoint_or_task_result(tmp_path):
    from langgraph.checkpoint.base import empty_checkpoint
    from app.infrastructure.persistence.graph_checkpointer import FencedSqliteSaver
    from app.infrastructure.persistence.sql.repositories import create_engine, SqlSessionStore
    from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
    from app.domain.session.ports.session_store import StaleSessionWrite
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    store = SqlSessionStore(engine)
    claim = await store.claim("s", buyer_id="b")
    token = ShoppingContext.set(ShoppingContextSnapshot("s", "b", "zh-CN", "CNY", session_fence=claim.fence))
    try:
        async with FencedSqliteSaver.from_conn_string(str(tmp_path / "graph.db")) as saver:
            saver.session_store = store
            config = {"configurable": {"thread_id": "s", "checkpoint_ns": ""}}
            saved = await saver.aput(config, empty_checkpoint(), {}, {})
            await store.claim("s", buyer_id="b")
            with pytest.raises(StaleSessionWrite):
                await saver.aput(saved, empty_checkpoint(), {}, {})
            with pytest.raises(StaleSessionWrite):
                await saver.aput_writes(saved, [("messages", "迟到的工具结果")], "task")
            assert (await saver.aget_tuple(config)).config == saved
            assert (await saver.aget_tuple(config)).pending_writes == []
    finally:
        ShoppingContext.reset(token)
        await engine.dispose()


async def test_native_search_keeps_tool_events_without_publishing_cards(tmp_path, monkeypatch):
    from ag_ui.core import RunAgentInput
    from app.presentation.ag_ui import parse_intent
    import json
    container = await _container(tmp_path, monkeypatch, ScriptedModel(mode="search"))
    body = RunAgentInput(thread_id="search", run_id="r",
        messages=[{"id": "u", "role": "user", "content": "查 P1003-S1"}],
        state={}, tools=[], context=[], forwarded_props={"buyerId": "buyer"})
    try:
        frames = [frame async for frame in run_frames(container.orchestrator, body, parse_intent(body))]
        events = [json.loads(line[6:]) for frame in frames for line in frame.splitlines() if line.startswith("data: ")]
        kinds = [event["type"] for event in events]
        assert kinds[-1] == "RUN_FINISHED"
        assert kinds.index("TOOL_CALL_START") < kinds.index("TOOL_CALL_END") < kinds.index("TOOL_CALL_RESULT")
        snapshots = [event["snapshot"] for event in events if event["type"] == "STATE_SNAPSHOT"]
        assert all("products" not in s for s in snapshots)
        assert snapshots[-1]["recommendation"] is snapshots[-1]["comparison"] is None
        receipt = next(e for e in events if e["type"]=="TOOL_CALL_RESULT")
        assert "P1003-S1" in receipt["content"]
    finally:
        await container.shutdown()


async def test_legacy_snapshot_is_preserved_without_silent_empty_graph(tmp_path, monkeypatch):
    import json
    container = await _container(tmp_path, monkeypatch, ScriptedModel())
    original = json.dumps({"context": [{"role": "user", "content": "历史内容"}]})
    try:
        store = container.session_store
        claim = await store.claim("old", buyer_id="buyer")
        await store.save_claim(claim, original)
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput("old", "buyer", "zh-CN", "CNY", "继续"),
            
        )
        assert result.error_code == "SESSION_VERSION_CHANGED"
        assert await store.load("old") == original
    finally:
        await container.shutdown()


async def test_native_graph_skill_catalog_reflects_deletion_and_buyer_scope(tmp_path, monkeypatch):
    model = ScriptedModel()
    container = await _container(tmp_path, monkeypatch, model)
    skills = container.orchestrator._sessions._main_factory.buyer_skill_store
    own = skills.save("buyer", "轻装方案", "选择轻便物品", "先核验重量再比较")
    skills.save("other", "他人私有方案", "不可展示", "他人的正文")
    try:
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput("skills", "buyer", "zh-CN", "CNY", "有什么方案"),
            
        )
        assert result.error is None
        directory = next(message.content for message in model.seen[-1] if message.name == "skill_catalog")
        assert "轻装方案" in directory and "他人私有方案" not in directory
        skills.delete("buyer", own["id"], own["version"])
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput("skills", "buyer", "zh-CN", "CNY", "现在呢"),
            
        )
        assert result.error is None
        directory = next(message.content for message in model.seen[-1] if message.name == "skill_catalog")
        assert "轻装方案" not in directory and directory.endswith("[]")
    finally:
        await container.shutdown()
