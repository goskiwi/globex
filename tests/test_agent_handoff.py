"""模块一：真实 LangGraph 主子图、结构化交付和交易证据，模型为确定性脚本。"""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from app.application.agents.main_agent import MainAgentFactory
from app.application.agents.handoff import build_submission_tool
from app.application.runtime.handoff import HandoffContext, HandoffMiddleware
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from app.application.agents.search_agent import SearchAgentFactory
from app.application.agents.trade_agent import TradeAgentFactory
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.throttle import GatewayThrottle
from tests.test_retrieval import _settings
from tests.trade_test_helpers import confirmation_env, test_address as sample_address


class ScriptedModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    cursor: int = 0

    @property
    def _llm_type(self):
        return "handoff-script"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if self.cursor >= len(self.responses):
            raise RuntimeError("脚本没有更多响应")
        # 每轮生成独立原生消息和调用 ID，重复脚本内容不能覆盖历史工具回执。
        response = self.responses[self.cursor].model_copy(deep=True, update={"id": None})
        self.cursor += 1
        response.tool_calls = [{**c, "id": f"{c['id']}-round-{self.cursor}"} for c in response.tool_calls]
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args, identifier="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identifier}])


def submission(**updates):
    return call("SubagentResult", {"status": "completed", "summary": "已完成候选研究", **updates}, "submit-1")


def task(**updates):
    return {"goal": "比较轻便背包", **updates}


@pytest.fixture(autouse=True)
def context():
    token = ShoppingContext.set(ShoppingContextSnapshot("handoff", "buyer", "zh-CN", "CNY"))
    try:
        yield
    finally:
        ShoppingContext.reset(token)


def factories(tmp_path, monkeypatch, responses, role="search", tools=None):
    model = ScriptedModel(responses=responses)
    module = f"app.application.agents.{role}_agent"
    monkeypatch.setattr(module + ".create_chat_model", lambda *a, **kw: model)
    settings = replace(_settings(tmp_path), harness_enabled=False, llm_fallback_model="")
    bus = TradeEventBus()
    registry, throttle = CircuitBreakerRegistry(), GatewayThrottle(5, 0)
    if role == "search":
        factory = SearchAgentFactory(settings, __import__('app.application.usecases.catalog_search', fromlist=['CatalogSearchUseCase']).CatalogSearchUseCase(None), bus, None, registry, throttle, model_client=None)
    else:
        factory = TradeAgentFactory(settings, None, None, None, bus, registry, throttle, model_client=None)
    factory.bind_harness(None)

    @tool
    def read_candidates() -> str:
        """读取测试候选。"""
        payload = {"tool": "product_search_tool", "result_ref": "ctx_demo", "hits": [
            {"product_id": "P1001", "skus": [{"sku_id": "P1001-S1"}]},
            {"product_id": "P1002", "skus": [{"sku_id": "P1002-S1"}]},
        ]}
        bus.publish("handoff", "tool.result", payload)
        return json.dumps(payload)

    factory.build_tools = lambda: tools if tools is not None else [read_candidates]
    dispatch = build_task_dispatch_tool(factory, factory, bus)
    return factory, model, dispatch, bus


async def test_parent_graph_injects_state_and_keeps_child_history_isolated(tmp_path, monkeypatch):
    candidate = {"product_id": "P1001", "sku_id": "P1001-S1", "reason": "符合要求"}
    search, child, dispatch, bus = factories(tmp_path, monkeypatch, [call("read_candidates", {}),
        submission(candidates=[candidate])])
    parent = ScriptedModel(responses=[call("update_shopping_state", {"update":{
        "filters":{"price_max_major":300,"target_currency":"CNY"}}}),call("task_dispatch", {
        "subagent_type": "search_agent", "task": task()}, "dispatch-1"), AIMessage(content="主 Agent 汇总完成")])
    monkeypatch.setattr("app.application.agents.main_agent.create_chat_model", lambda *a, **kw: parent)
    trade = SimpleNamespace(build_tools=lambda: [], bind_harness=lambda *args: None)
    from tests.test_ag_ui import EmptyPreferences
    main_factory = MainAgentFactory(search._settings, search, trade, bus, EmptyPreferences(),
        CircuitBreakerRegistry(), GatewayThrottle(5, 0), checkpointer=InMemorySaver(), model_client=None)
    graph = main_factory.build().graph
    state = await graph.ainvoke({"messages": [HumanMessage(content="无关旧话题", name="other"),
        HumanMessage(content="预算300元，寄到中国。这次可以接受塑料。", name="buyer")]},
        config={"configurable": {"thread_id": "handoff"}})
    sent = json.loads(next(m.content for m in child.seen[0] if isinstance(m, HumanMessage)))
    assert sent["parent_context"]["effective_search"]["parameters"]["price_max_major"] == 300
    assert "这次可以接受塑料" in sent["parent_context"]["latest_user_request"]
    assert "无关旧话题" not in json.dumps(sent, ensure_ascii=False)
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "task_dispatch")
    delivered = json.loads(result.content)["candidates"][0]
    assert {k:v for k,v in delivered.items() if k != "facts"} == {**candidate, "unmet_constraints": []}
    assert delivered["facts"]["result_ref"] == "ctx_demo"
    assert json.loads(result.content)["status"] == "completed"
    assert not any(isinstance(m, ToolMessage) and m.name == "read_candidates" for m in state["messages"])
    schema = convert_to_openai_tool(as_langchain_tool(dispatch))["function"]["parameters"]["properties"]
    assert set(schema) == {"subagent_type", "task"}


@pytest.mark.parametrize("response", [AIMessage(content="我已经完成了"),
    submission(status="partial"), submission(status="needs_input"),
    submission(status="completed", questions=["预算多少？"]), submission()])
async def test_invalid_or_unsupported_completion_is_failed(tmp_path, monkeypatch, response):
    _, _, dispatch, _ = factories(tmp_path, monkeypatch, [response])
    result = await dispatch("search_agent", task())
    assert not result.ok
    assert result.data["status"] == "failed"


@pytest.mark.parametrize("updates", [
    {"candidates": [{"product_id": "NOT-IN-CATALOG", "reason": "推荐"}]},
    {"candidates": [{"product_id": "P1001", "sku_id": "P1002-S1", "reason": "推荐"}]},
    {"evidence_refs": ["ctx_invented"]},
])
async def test_result_must_match_actual_child_evidence(tmp_path, monkeypatch, updates):
    _, _, dispatch, _ = factories(tmp_path, monkeypatch, [call("read_candidates", {}), submission(**updates)])
    result = await dispatch("search_agent", task())
    assert result.data["status"] == "failed"
    assert result.data["evidence_refs"] == ["ctx_demo"]


async def test_nonblocking_notes_do_not_turn_completed_task_into_failure(tmp_path, monkeypatch):
    _, _, dispatch, _ = factories(tmp_path, monkeypatch, [call("read_candidates", {}),
        submission(summary="已完成；未提供用户本次未要求的容量数据")])
    result = (await dispatch("search_agent", task())).data
    assert result["status"] == "completed"
    assert "未提供用户本次未要求的容量数据" in result["summary"]


@pytest.mark.parametrize("updates", [
    {"status": "needs_input", "questions": ["收货地址是什么？"]},
    {"status": "partial", "unmet_constraints": ["尚未找到防水证明"]},
    {"status": "failed", "issues": ["检索服务不可用"]},
])
async def test_explicit_noncompleted_statuses_survive_handoff(tmp_path, monkeypatch, updates):
    _, _, dispatch, _ = factories(tmp_path, monkeypatch, [submission(**updates)])
    result = (await dispatch("search_agent", task())).data
    assert result["status"] == updates["status"]
    assert result[next(k for k in updates if k != "status")] == updates[next(k for k in updates if k != "status")]


async def test_pending_confirmation_never_becomes_executed(tmp_path, monkeypatch):
    factory, _, dispatch, bus = factories(tmp_path, monkeypatch, [call("prepare_trade", {}),
        submission(summary="订单已经支付发货")], role="trade")

    @tool
    def prepare_trade() -> str:
        """准备测试确认单，不执行交易。"""
        payload = {"tool": "create_order_tool", "confirmation": {
            "confirmation_id": "confirm-1", "action": "create", "status": "pending",
            "payload": {"items": [{"product_id": "P1001", "sku_id": "P1001-S1"}]}}}
        bus.publish("handoff", "tool.result", payload)
        return json.dumps(payload)

    factory.build_tools = lambda: [prepare_trade]
    result = (await dispatch("trade_agent", task())).data
    assert result["status"] == "completed"
    assert result["transaction_state"] == "awaiting_confirmation"
    assert "尚未执行" in result["summary"] and "已经支付" not in result["summary"]
    assert result["trade_results"][0]["confirmation_id"] == "confirm-1"


async def test_context_owner_mismatch_does_not_execute_child(tmp_path, monkeypatch):
    _, model, dispatch, _ = factories(tmp_path, monkeypatch, [submission()])
    runtime = SimpleNamespace(tool_call_id="d", state={"shopping_work_owner": "other"})
    result = (await dispatch("search_agent", task(), runtime)).data
    assert result["status"] == "failed" and not model.seen


async def test_cancellation_is_not_swallowed():
    class Factory:
        def build(self):
            return self

        async def ainvoke(self, inputs, **kwargs):
            raise asyncio.CancelledError()

    dispatch = build_task_dispatch_tool(Factory(), Factory(), TradeEventBus())
    with pytest.raises(asyncio.CancelledError):
        await dispatch("search_agent", task())


async def test_simple_request_does_not_require_dispatch(tmp_path, monkeypatch):
    factory, child, dispatch, _ = factories(tmp_path, monkeypatch, [])
    parent = ScriptedModel(responses=[call("read_candidates", {}), AIMessage(content="已找到候选")])
    graph = create_agent(parent, tools=[as_langchain_tool(dispatch), *factory.build_tools()])
    result = await graph.ainvoke({"messages": [HumanMessage(content="找个背包", name="buyer")]})
    assert not child.seen
    assert any(isinstance(m, ToolMessage) and m.name == "read_candidates" for m in result["messages"])


async def test_parallel_results_do_not_share_evidence_and_failure_keeps_other_result():
    bus = TradeEventBus()
    ready = asyncio.Event()

    @tool
    async def read_scope(runtime: ToolRuntime[HandoffContext]) -> str:
        """读取当前子任务的测试候选。"""
        job = runtime.state["handoff_task"]["goal"]
        if job == "success":
            bus.publish("handoff", "tool.result", {"tool": "product_search_tool",
                "hits": [{"product_id": "P1001", "skus": []}], "result_ref": "ctx_success"})
            ready.set()
        else:
            await ready.wait()
        return "当前调用结束"

    class Factory:
        def build(self):
            answer = submission(candidates=[{"product_id": "P1001", "reason": "候选"}])
            return create_agent(ScriptedModel(responses=[call("read_scope", {}), answer, answer, answer]),
                tools=[read_scope, build_submission_tool()], middleware=[HandoffMiddleware()],
                context_schema=HandoffContext)

    dispatch = build_task_dispatch_tool(Factory(), Factory(), bus)
    good, bad = await asyncio.gather(dispatch("search_agent", task(goal="success")),
                                     dispatch("search_agent", task(goal="failed")))
    assert good.data["status"] == "completed"
    assert bad.data["status"] == "failed"
    assert bad.data["evidence_refs"] == []


@pytest.mark.parametrize('kind',['products','product_details','product_view'])
async def test_historical_lookup_is_valid_but_labelled_historical(tmp_path, monkeypatch, kind):
    from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
    factory, _, dispatch, _ = factories(tmp_path, monkeypatch, [])
    ref = await factory.evidence_store.save("buyer", "handoff", kind, {
        "hits": [{"product_id": "P1001", "title": "合成背包", "skus": [{"sku_id": "P1001-S1"}]}]})
    model = ScriptedModel(responses=[call("conversation_fact_lookup", {"result_ref": ref}),
        submission(candidates=[{"product_id": "P1001", "reason": "历史候选"}])])
    monkeypatch.setattr("app.application.agents.search_agent.create_chat_model", lambda *a, **kw: model)
    factory.build_tools = lambda: [as_langchain_tool(build_conversation_fact_lookup(factory.evidence_store))]
    result = (await dispatch("search_agent", task(evidence_refs=[ref]))).data
    assert result["status"] == "completed"
    assert result["historical_evidence_refs"] == [ref]


@pytest.mark.parametrize("valid_submission", [True, False])
@pytest.mark.parametrize("harness_enabled", [True, False])
async def test_real_trade_tools_prepare_without_order_or_inventory_write(confirmation_env, tmp_path, monkeypatch, valid_submission, harness_enabled):
    from dataclasses import asdict
    from app.application.usecases.order_usecases import PlaceOrderUseCase, QueryOrderUseCase, CancelOrderUseCase
    env = confirmation_env
    product = (await env.products.list_all())[0]
    sku = product.skus[0]
    from tests.shopping_state_helpers import selected_lines
    ShoppingContext.set(replace(ShoppingContext.current(), selected_lines=selected_lines(sku.sku_id,quantity=2)))
    await env.evidence.save("buyer", "handoff", "products", {"hits": [{
        "product_id": product.product_id, "default_sku_id": sku.sku_id,
        "skus": [{"sku_id": sku.sku_id}]}]})
    result_messages = [submission(summary="确认单准备完成")] if valid_submission else [submission(status="partial") for _ in range(3)]
    model = ScriptedModel(responses=[call("create_order_tool", {
        "sku_ids": [sku.sku_id], "shipping_address": asdict(sample_address())}),
        *result_messages])
    monkeypatch.setattr("app.application.agents.trade_agent.create_chat_model", lambda *a, **kw: model)
    factory = TradeAgentFactory(replace(_settings(tmp_path), harness_enabled=harness_enabled),
        PlaceOrderUseCase(env.service), QueryOrderUseCase(env.store), CancelOrderUseCase(env.service),
        env.bus, CircuitBreakerRegistry(), GatewayThrottle(3, 0), model_client=None)
    factory.evidence_store = env.evidence
    factory.bind_harness(None)
    dispatch = build_task_dispatch_tool(factory, factory, env.bus)
    before = await env.store.get_inventory([sku.sku_id])
    result = (await dispatch("trade_agent", task())).data
    assert result["status"] == ("completed" if valid_submission else "failed")
    assert result["transaction_state"] == "awaiting_confirmation"
    confirmation = await env.service.get(result["trade_results"][0]["confirmation_id"], "buyer", "handoff")
    assert confirmation["confirmation"]["status"] == "pending"
    assert confirmation["confirmation"]["payload"]["items"][0]["quantity"] == 2
    assert confirmation["confirmation"].get("result") is None
    assert await env.store.get_inventory([sku.sku_id]) == before
