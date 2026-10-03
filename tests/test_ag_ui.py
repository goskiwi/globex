# -*- coding: utf-8 -*-
"""AG-UI 合同测试：原生事件夹具替代付费模型，商品查询仍走真实 UseCase。"""
from __future__ import annotations

from tests.ag_ui_runtime_helpers import run_frames
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from ag_ui.core import RunAgentInput
from app.application.runtime.events import ToolEvent
from app.application.runtime.results import ToolResultState
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr
from app.application.runtime.tools import as_langchain_tool

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.presentation.ag_ui import parse_intent, register_ag_ui_routes


def request_data(**changes):
    data = {
        "threadId": "session-test", "runId": "run-test", "state": {},
        "messages": [
            {"id": "old-user", "role": "user", "content": "你好"},
            {"id": "old-assistant", "role": "assistant", "content": "需要什么商品？"},
            {"id": "new-user", "role": "user", "content": "旅行三件套"},
        ],
        "tools": [], "context": [], "forwardedProps": {"buyerId": "buyer-test"},
    }
    data.update(changes)
    return data


def decode_frames(frames):
    return [json.loads(line[6:]) for frame in frames for line in frame.splitlines() if line.startswith("data: ")]


class EmptyPreferences:
    async def list_by_buyer(self, buyer_id):
        return []


class Sessions:
    def __init__(self, agent):
        self.agent = agent
        self.persisted = 0

    async def get_or_create(self, session_id):
        self.agent.config = {"configurable": {"thread_id": session_id}}
        return self.agent

    async def persist(self, session_id):
        self.persisted += 1

    async def invalidate(self, session_id):
        self.invalidated = session_id


class ScriptedAgent:
    """脚本只决定模型会发哪些调用，商品与价格不使用测试伪造数据。"""

    name = "CatalogTestAgent"

    def __init__(self, bus, *, fail=False, block=False, swallow_cancel=False, query="旅行三件套"):
        self.bus = bus
        self.fail = fail
        self.block = block
        self.swallow_cancel = swallow_cancel
        self.query = query
        self.state = SimpleNamespace(summary=None, context=[])
        self.closed = asyncio.Event()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.usecase = CatalogSearchUseCase(InMemoryProductRepository())
        owner = self
        class Model(BaseChatModel):
            @property
            def _llm_type(self):
                return "agui-scripted-model"
            def bind_tools(self, tools, **kwargs):
                return self
            def _generate(self, *args, **kwargs):
                raise NotImplementedError("测试仅执行异步路径")
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                if isinstance(messages[-1], ToolMessage):
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(
                        content="这是根据商品库找到的结果。"))])
                owner.calls += 1
                owner.in_flight += 1
                owner.peak = max(owner.peak, owner.in_flight)
                owner.started.set()
                try:
                    if run_manager:
                        await run_manager.on_llm_new_token("正在查找商品。")
                    if owner.block:
                        await owner.release.wait()
                    if owner.fail:
                        raise ValueError("测试执行失败")
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(
                        content="", tool_calls=[{"id": "search-" + str(owner.calls),
                            "name": "product_search_tool",
                            "args": {"normalized_query": owner.query}}]))])
                except asyncio.CancelledError:
                    if not owner.swallow_cancel:
                        raise
                    return ChatResult(generations=[ChatGeneration(
                        message=AIMessage(content="本轮已中断"),
                        generation_info={"finish_reason": "interrupted"})])
                finally:
                    owner.in_flight -= 1
                    owner.closed.set()
        tool = build_product_search_tool(self.usecase, bus)
        from app.application.runtime.working_state import WorkingStateMiddleware
        from app.application.runtime.middleware import BusinessToolMiddleware
        self.graph = create_agent(Model(), tools=[as_langchain_tool(tool)], middleware=[BusinessToolMiddleware(None,bus),WorkingStateMiddleware()], checkpointer=InMemorySaver())
        self.config = {}
        self.tool_names = frozenset({"product_search_tool"})

def make_orchestrator(**kwargs):
    bus = TradeEventBus()
    agent = ScriptedAgent(bus, **kwargs)
    sessions = Sessions(agent)
    orchestrator = MainAgentOrchestrator(sessions, bus)
    return orchestrator, agent, sessions


async def collect(orchestrator, **changes):
    body = RunAgentInput.model_validate(request_data(**changes))
    frames = [frame async for frame in run_frames(orchestrator, body, parse_intent(body))]
    return decode_frames(frames)


async def test_search_stays_internal_and_preserves_final_message_history():
    orchestrator, agent, sessions = make_orchestrator()
    events = await collect(orchestrator)
    types = [event["type"] for event in events]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert types.index("TOOL_CALL_END") < types.index("TOOL_CALL_RESULT")
    result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
    search = json.loads(result["content"])
    final_state = [event["snapshot"] for event in events if event["type"] == "STATE_SNAPSHOT"][-1]
    assert search["hits"] and "image_url" not in search["hits"][0]
    assert "products" not in final_state and "searchCompleted" not in final_state
    assert final_state["recommendation"] is final_state["comparison"] is None
    assert final_state["status"] == "completed"
    assert final_state["process"]['steps'][0]["status"] == "completed"
    assert 'progress' not in final_state
    final_messages = next(event["messages"] for event in events if event["type"] == "MESSAGES_SNAPSHOT")
    assert final_messages[0]["id"] == "new-user"
    assert not any(m["id"] in {"old-user","old-assistant"} for m in final_messages), "客户端历史不能伪造服务器账本"
    assert final_messages[-1]["content"] == "这是根据商品库找到的结果。"
    assert sessions.persisted == 1


async def test_empty_search_clears_client_products():
    orchestrator, _, _ = make_orchestrator(query="quantum flux capacitor")
    events = await collect(orchestrator, state={"products": [{"product_id": "fake-old-product"}]})
    snapshots = [event["snapshot"] for event in events if event["type"] == "STATE_SNAPSHOT"]
    assert all("products" not in s and "searchCompleted" not in s for s in snapshots)
    assert snapshots[-1]["recommendation"] is None


async def test_both_entrypoints_execute_graph_instead_of_text_only_cache():
    orchestrator, agent, _ = make_orchestrator()
    cache = SimpleNamespace(
        lookup=AsyncMock(return_value=SimpleNamespace(reply="缓存推荐文本", similarity=1.0, matched_query="旅行三件套")),
        remember=AsyncMock(),
    )
    orchestrator._semantic_cache = cache
    events = await collect(orchestrator)
    assert agent.calls == 1
    assert any(event["type"] == "TOOL_CALL_RESULT" for event in events)
    cache.lookup.assert_not_awaited()
    cache.remember.assert_not_awaited()
    body = RunAgentInput.model_validate(request_data(threadId="fresh-cache-session"))
    result = await orchestrator.handle_intent(parse_intent(body))
    assert result.final_text != "缓存推荐文本" and agent.calls == 2
    cache.lookup.assert_not_awaited()


@pytest.mark.parametrize("reason", ["completed","interrupted","error","exceed_max_iters"])
def test_native_finished_reason_enum_is_recognized(reason):
    adapter = AGUIRunAdapter(RunAgentInput.model_validate(request_data()), lambda event: None)
    adapter.on_agent_event(SimpleNamespace(type="REPLY_END", finished_reason=reason))
    expected_error = str(reason).lower() in {"error", "interrupted", "exceed_max_iters"}
    assert (adapter.error is not None) == expected_error


@pytest.mark.parametrize("state", list(ToolResultState))
def test_native_tool_result_state_enum_is_recognized(state):
    adapter = AGUIRunAdapter(RunAgentInput.model_validate(request_data()), lambda event: None)
    adapter.on_agent_event(ToolEvent("TOOL_RESULT_END", tool_call_id="call", state=state))
    assert adapter.state["process"]['steps'][-1]["status"] == (
        "completed" if state == ToolResultState.SUCCESS else "failed"
    )


async def test_failure_closes_message_and_ends_with_run_error():
    orchestrator, agent, sessions = make_orchestrator(fail=True)
    events = await collect(orchestrator)
    types = [event["type"] for event in events]
    assert types[-1] == "RUN_ERROR"
    assert "RUN_FINISHED" not in types
    assert types.count("TEXT_MESSAGE_START") == types.count("TEXT_MESSAGE_END") == 0
    assert agent.closed.is_set() and sessions.persisted == 1


def test_parallel_same_named_tools_keep_distinct_results():
    events = []
    adapter = AGUIRunAdapter(RunAgentInput.model_validate(request_data()), events.append)
    for identifier in ["call-a", "call-b"]:
        adapter.on_agent_event(ToolEvent("TOOL_CALL_START", tool_call_id=identifier, tool_call_name="product_search_tool"))
        adapter.on_agent_event(ToolEvent("TOOL_CALL_END", tool_call_id=identifier))
        adapter.on_agent_event(ToolEvent("TOOL_RESULT_START", tool_call_id=identifier, tool_call_name="product_search_tool"))
    for identifier in ["call-b", "call-a"]:
        adapter.on_agent_event(ToolEvent("TOOL_RESULT_TEXT_DELTA", tool_call_id=identifier, delta=identifier))
        adapter.on_agent_event(ToolEvent("TOOL_RESULT_END", tool_call_id=identifier, state=ToolResultState.SUCCESS))
    results = [event for event in events if event.type == "TOOL_CALL_RESULT"]
    assert [(event.tool_call_id, event.content) for event in results] == [
        ("run-test:tool:call-b", "call-b"), ("run-test:tool:call-a", "call-a"),
    ]


async def test_same_session_legacy_and_ag_ui_runs_are_serialized():
    orchestrator, agent, sessions = make_orchestrator(block=True)
    body = RunAgentInput.model_validate(request_data())
    first = asyncio.create_task(orchestrator.handle_intent(parse_intent(body)))
    await agent.started.wait()
    second = asyncio.create_task(collect(orchestrator))
    await asyncio.sleep(0.02)
    assert agent.calls == 1
    agent.release.set()
    await first
    await second
    assert agent.peak == 1 and sessions.persisted == 2


async def test_event_observers_are_task_scoped_including_children():
    bus = TradeEventBus()
    seen = [[], []]

    async def run(index):
        with observe_run_events(seen[index].append):
            async def child():
                await asyncio.sleep(0)
                bus.publish("same-session", "tool.result", {"owner": index})
            await asyncio.create_task(child())

    await asyncio.gather(run(0), run(1))
    assert [[event.payload["owner"] for event in events] for events in seen] == [[0], [1]]


async def test_http_endpoint_contract_and_reject_stale_resume(tmp_path):
    orchestrator, _, _ = make_orchestrator()
    app = FastAPI()
    from app.presentation.ag_ui_runtime import AGUIRuntime
    from app.infrastructure.ag_ui_journal import AGUIJournal
    runtime=AGUIRuntime(AGUIJournal(tmp_path/"journal.db"),orchestrator)
    await runtime.startup()
    register_ag_ui_routes(app, lambda: orchestrator, get_runtime=lambda:runtime)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/commerce/ag-ui/run", json=request_data())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert decode_frames([response.text])[-1]["type"] == "RUN_FINISHED"
        response = await client.post("/commerce/ag-ui/run", json=request_data(
            runId="resume-test", resume=[{"interruptId": "unavailable", "status": "resolved", "payload": {"approved": True}}],
        ))
        assert response.status_code == 409  # 未知暂停点在持久运行入口被拒绝
    await runtime.shutdown()


@pytest.mark.parametrize("swallow_cancel", [False, True])
async def test_http_disconnect_preserves_run_until_explicit_cancel(swallow_cancel, tmp_path):
    orchestrator, agent, sessions = make_orchestrator(block=True, swallow_cancel=swallow_cancel)
    conversations = SimpleNamespace(touch_session=AsyncMock(), append_turn=AsyncMock(), append_events=AsyncMock())
    orchestrator._conversation_store = conversations
    app = FastAPI()
    from app.presentation.ag_ui_runtime import AGUIRuntime
    from app.infrastructure.ag_ui_journal import AGUIJournal
    runtime=AGUIRuntime(AGUIJournal(tmp_path/"journal.db"),orchestrator)
    await runtime.startup()
    register_ag_ui_routes(app, lambda: orchestrator, get_runtime=lambda:runtime)
    disconnected = asyncio.Event()
    received_body = False
    responses = []

    async def receive():
        nonlocal received_body
        if not received_body:
            received_body = True
            return {"type": "http.request", "body": json.dumps(request_data()).encode(), "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        responses.append(message)
        if b"RUN_STARTED" in message.get("body", b""):
            disconnected.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/commerce/ag-ui/run", "raw_path": b"/commerce/ag-ui/run",
        "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234), "server": ("test", 80), "root_path": "",
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    assert disconnected.is_set() and not agent.closed.is_set()
    await asyncio.wait_for(agent.started.wait(),2)
    await runtime.cancel("run-test", "buyer-test")
    await runtime.shutdown()
    assert agent.closed.is_set()
    assert agent.in_flight == 0 and sessions.persisted == 0  # 已撤销写入租约，不落无效 checkpoint
    saved_turns = [call.args[0] for call in conversations.append_turn.await_args_list]
    assert not saved_turns
    stopped = await runtime.journal.run("run-test", "buyer-test")
    assert stopped["status"] == "stopped"
    assert not orchestrator._session_locks["session-test"].locked()
    assert not any(b"RUN_FINISHED" in message.get("body", b"") for message in responses)

@pytest.mark.parametrize('kind',['capability','contract'])
async def test_old_session_version_returns_actionable_protocol_error(kind):
    from app.infrastructure.capability_registry import CapabilityVersionChanged
    from app.infrastructure.prompt_registry import PromptContractChanged
    orchestrator,agent,sessions=make_orchestrator()
    error=CapabilityVersionChanged('old') if kind=='capability' else PromptContractChanged('old')
    sessions.get_or_create=AsyncMock(side_effect=error)
    frames=[frame async for frame in run_frames(orchestrator,RunAgentInput.model_validate(request_data()),parse_intent(RunAgentInput.model_validate(request_data())))]
    last=decode_frames(frames)[-1]
    assert last['type']=='RUN_ERROR' and last['code']=='SESSION_VERSION_CHANGED'
    assert '旧记录仍保留' in last['message']
    assert agent.calls==0


async def test_context_capacity_error_is_actionable_and_never_retries_business():
    from app.application.runtime.errors import ContextCapacityError
    orchestrator,agent,sessions=make_orchestrator()
    orchestrator._reply=AsyncMock(side_effect=ContextCapacityError('unsafe input'))
    body=RunAgentInput.model_validate(request_data())
    frames=[frame async for frame in run_frames(orchestrator,body,parse_intent(body))]
    last=decode_frames(frames)[-1]
    assert last['type']=='RUN_ERROR' and last['code']=='CONTEXT_CAPACITY_EXCEEDED'
    assert '分批比较' in last['message'] and '原始记录已保留' in last['message']
    assert agent.calls==0
    orchestrator._reply.assert_awaited_once()
