"""预算、持久证据和压缩的行为回归。"""
from app.application.runtime.execution import ExecutionResult
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from app.infrastructure.budget import init_budget
from app.application.runtime.errors import ExecutionStopped
import httpx
from tests.native_model_helpers import client_model, completion
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore, product_decision_view
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup


class BudgetWire:
    def __init__(self):
        self.calls = 0
        self.wait = asyncio.Event()
        self.started = asyncio.Event()
    async def __call__(self, request):
        self.calls += 1
        self.started.set()
        await self.wait.wait()
        raw=completion()
        raw['usage']={'prompt_tokens':70,'completion_tokens':10,'total_tokens':80}
        return httpx.Response(200,json=raw)


async def test_exhaustion_never_enters_upstream_or_throttle(tmp_path):
    budget = init_budget(1000)
    budget.charge("previous", 999)
    wire=BudgetWire()
    model = await client_model(tmp_path,wire)
    try:
        with pytest.raises(ExecutionStopped, match='预算不足'):
            await asyncio.wait_for(model.ainvoke('查询'), 2)
        assert wire.calls == 0
        assert budget.reserved == 0
    finally:
        init_budget(0)
        await model.aclose()


async def test_parallel_reservations_do_not_spend_same_remainder(tmp_path):
    budget = init_budget(1100)
    wire=BudgetWire()
    model = await client_model(tmp_path,wire)
    try:
        first = asyncio.create_task(model.ainvoke('查询'))
        await asyncio.wait_for(wire.started.wait(),2)
        assert budget.reserved == 1100
        with pytest.raises(ExecutionStopped):
            await model.ainvoke('查询')
        assert wire.calls == 1
        wire.wait.set()
        await first
        assert budget.used == 80 and budget.reserved == 0
        assert (await model.ainvoke('查询')).text=='OK'
        assert wire.calls == 2
    finally:
        wire.wait.set()
        init_budget(0)
        await model.aclose()


def test_budget_does_not_own_business_results_or_generate_replies():
    from app.infrastructure import budget
    assert not hasattr(budget, 'remember_verified_result')
    assert not hasattr(budget, 'rule_fallback_text')
    assert not hasattr(budget.TokenBudget(1000), 'fallback_used')


async def test_evidence_survives_reopen_and_rejects_cross_scope(tmp_path):
    store = ContextEvidenceStore(tmp_path / "evidence.db")
    ref = await store.save("buyer", "session", "products", {"hits": [{"product_id": "P1"}]})
    reopened = ContextEvidenceStore(store.path)
    assert (await reopened.get("buyer", "session", ref))["data"]["hits"][0]["product_id"] == "P1"
    assert await reopened.get("other", "session", ref) is None
    assert await reopened.get("buyer", "other", ref) is None


async def test_lookup_second_item_from_latest_display_after_restart(tmp_path):
    store = ContextEvidenceStore(tmp_path / "evidence.db")
    await store.save("b", "s", "display_batch", {"hits": [{"product_id": "OLD"}]})
    await store.save("b", "s", "display_batch", {"hits": [{"product_id": "P1"}, {"product_id": "P2"}]})
    token = ShoppingContext.set(ShoppingContextSnapshot("s", "b", "zh-CN", "CNY"))
    try:
        result = await build_conversation_fact_lookup(ContextEvidenceStore(store.path))(batch=0, position=2)
        assert json.loads(result.text)["records"][0]["data"]["hits"] == [{"product_id": "P2"}]
        await store.save("b", "s", "display_batch", {"hits": []})
        result = await build_conversation_fact_lookup(store)(batch=0, position=2)
        assert json.loads(result.text)["records"][0]["data"]["hits"] == []
    finally:
        ShoppingContext.reset(token)


def test_decision_projection_keeps_exact_prices_skus_but_removes_media():
    full = {"hits": [{"product_id": "P1", "price_major": 19.98, "currency": "CNY", "skus": [{"sku_id": "S1", "stock": 3}], "description": "x"*1000, "image_url": "https://image"}], "result_ref": "ctx_1"}
    view = product_decision_view(full)
    assert view["hits"][0]["skus"] == full["hits"][0]["skus"]
    assert view["hits"][0]["price_major"] == 19.98
    assert "description" not in view["hits"][0] and "image_url" not in view["hits"][0]
    assert "description" in full["hits"][0]


async def test_old_tool_archive_runs_before_summary_with_tool_call_pair_intact(tmp_path):
    from tests.native_context_helpers import policy,apply
    from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
    messages=[HumanMessage(name='b',content='旧请求'),AIMessage(content='',tool_calls=[{'id':'call','name':'large_tool','args':{}}]),
        ToolMessage(id='result',tool_call_id='call',name='large_tool',content='证据'*1000)]
    messages.extend(HumanMessage(name='b',content='近轮') for _ in range(6))
    middleware=policy(tmp_path,product_tokens=1)
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        updates=await middleware.compact_checkpoint({'messages':messages,'read_tool_messages':['result']})
        assert len(updates['messages'])==1 and updates['messages'][0].tool_call_id=='call'
        ref=json.loads(updates['messages'][0].content)['result_ref']
        saved=await middleware.store.get('b','s',ref)
        assert saved['data']['text']=='证据'*1000
        middleware.model.ainvoke.assert_not_awaited()
    finally:ShoppingContext.reset(token)


async def test_delayed_old_fence_cannot_replace_latest_candidate(tmp_path):
    store = ContextEvidenceStore(tmp_path / "evidence.db")
    token = ShoppingContext.set(ShoppingContextSnapshot("s", "b", "zh-CN", "CNY", session_fence=2))
    try:
        await store.save("b", "s", "products", {"hits": [{"product_id": "NEW"}]})
        ShoppingContext.set_session_fence(1)
        await store.save("b", "s", "products", {"hits": [{"product_id": "OLD"}]})
        latest = (await store.search("b", "s", kind="products", limit=1))[0]
        assert latest["data"]["hits"][0]["product_id"] == "NEW"
    finally:
        ShoppingContext.reset(token)


async def test_retry_cannot_bypass_exhausted_budget(tmp_path,monkeypatch):
    from dataclasses import replace
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import HumanMessage
    from app.application.runtime.middleware import GatewayModelMiddleware
    from tests.test_retrieval import _settings
    budget = init_budget(1100)
    calls=[]
    def upstream(request):
        calls.append(request)
        return httpx.Response(429,json={'error':{'message':'429 rate limit exceeded','type':'error'}})
    model=await client_model(tmp_path,upstream)
    middleware=GatewayModelMiddleware(replace(_settings(tmp_path),llm_fallback_model='',llm_max_retries=2),model.gateway, client=None)
    monkeypatch.setattr('app.application.runtime.middleware.asyncio.sleep',AsyncMock())
    request=ModelRequest(model=model,messages=[HumanMessage(content='查询')],tools=[],state={},runtime=None)
    async def invoke(request):return await request.model.ainvoke(request.messages)
    try:
        with pytest.raises(ExecutionStopped):
            await middleware.awrap_model_call(request,invoke)
        assert len(calls)==1
        assert budget.used == 1100 and budget.reserved == 0
    finally:
        init_budget(0)
        await model.aclose()


async def test_preference_withdrawal_is_authoritative_after_restart(tmp_path):
    from app.application.memory.preference_resolution import resolve_preferences
    from app.application.memory.preference_selector import PreferenceSelector
    from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore
    from app.domain.buyer.preference import BuyerPreference
    store = JsonFilePreferenceStore(tmp_path / "preferences")
    await store.append(BuyerPreference("buyer", "dislike", "不要塑料材质"))
    first = await resolve_preferences(store, PreferenceSelector(), "buyer", "推荐背包", 5)
    assert "不要塑料材质" in first.hint and first.facts
    await store.delete("buyer", "不要塑料材质")
    second = await resolve_preferences(JsonFilePreferenceStore(tmp_path / "preferences"),
                                       PreferenceSelector(), "buyer", "推荐背包", 5)
    assert "历史已撤回偏好不得恢复" in second.hint and not second.facts


async def test_preference_read_failure_stops_state_preparation():
    from app.application.memory.preference_resolution import resolve_preferences
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    store = SimpleNamespace(list_by_buyer=AsyncMock(side_effect=RuntimeError("database error")))
    with pytest.raises(RuntimeError, match="偏好读取失败"):
        await resolve_preferences(store, None, "buyer", "耳机", 5)


async def test_explicit_comparison_order_survives_persistence_and_restart(tmp_path):
    from tests.test_ag_ui import make_orchestrator
    from app.application.agents.orchestrator import SubmitIntentInput
    from app.domain.catalog.product_search_spec import ProductSearchSpec
    orchestrator, agent, _ = make_orchestrator()
    store = ContextEvidenceStore(tmp_path/'candidate.db')
    orchestrator._evidence_store = store
    async def reply(session, current_agent, inputs):
        # 检索完成顺序不决定展示；明确提交的比较顺序才进入展示批次。
        cards=[]
        for identifier in ['P1003-S1','P1001-S2']:
            result = await agent.usecase.execute(ProductSearchSpec(normalized_query=identifier))
            cards.extend(result['hits'])
            orchestrator._bus.publish(session, 'tool.result', {'tool':'product_search_tool', **result})
        orchestrator._bus.publish(session, 'comparison.result', {'hits':list(reversed(cards))})
        return ExecutionResult('已完成对比', 'completed', product_delivery_complete=True)
    orchestrator._reply = reply
    await orchestrator.handle_intent(SubmitIntentInput('session-test','buyer-test','zh-CN','CNY','对比P1001-S2和P1003-S1'))
    restarted = ContextEvidenceStore(store.path)
    latest = (await restarted.search('buyer-test','session-test',kind='display_batch',limit=1))[0]
    assert [h['product_id'] for h in latest['data']['hits']] == ['P1001','P1003']
    token = ShoppingContext.set(ShoppingContextSnapshot('session-test','buyer-test','zh-CN','CNY'))
    try:
        chunk = await build_conversation_fact_lookup(restarted)(batch=0, position=2)
        payload = json.loads(chunk.text)
        assert 'P1003-S1' in json.dumps(payload)
    finally:
        ShoppingContext.reset(token)
