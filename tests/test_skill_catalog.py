"""原生图/HTTP/SQLite 验证 Skill 目录；目录按请求投影，不再依赖旧 SDK 水位。"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage

from app.application.agents.orchestrator import SubmitIntentInput
from app.application.runtime.skills import skill_directory
from app.infrastructure.buyer_skills import BuyerSkillConflict
from app.infrastructure.capability_registry import CapabilityRegistry, CapabilityVersionChanged
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.domain.session.ports.session_store import StaleSessionWrite
from tests.test_capability_registry import publish, document
from tests.native_model_helpers import client_model, completion
from tests.test_langgraph_runtime import _container, ScriptedModel, PreferenceStore


@pytest.fixture
async def case(tmp_path, monkeypatch):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion())
    model = await client_model(tmp_path, handler)
    container = await _container(tmp_path / "runtime", monkeypatch, model)
    factory = container.orchestrator._sessions._main_factory
    factory.skill_catalog_mode = "append_only"
    async def ask(text, buyer="buyer"):
        return await container.orchestrator.handle_intent(
            SubmitIntentInput("session", buyer, "zh-CN", "CNY", text),
            )
    async def state():
        session = container.orchestrator._sessions._agents["session"]
        return await session.graph.aget_state(session.config)
    value = SimpleNamespace(personal=factory.buyer_skill_store, registry=factory.capability_registry,
        requests=requests, model=model, container=container, ask=ask, state=state)
    try:
        yield value
    finally:
        await container.shutdown()
        await model.aclose()


def catalogue(case):
    messages = [m for m in case.requests[-1]["messages"] if m.get("name") == "skill_catalog"]
    assert len(messages) == 1
    return json.loads(messages[0]["content"].rsplit("\n", 1)[-1])


def conversation(messages):
    # 服务端当前事实提示每轮刷新；用户/助手原消息不得随目录变化改写。
    return [message for message in messages if message.name not in {
        "memory_hint", "trade_state", "candidate_state", "skill_catalog",
    }]


async def test_unchanged_catalog_is_injected_once_and_http_prefix_is_immutable(case):
    case.personal.save("buyer", "背包方案", "通勤", "轻便优先")
    assert not (await case.ask("选背包")).error
    before = deepcopy(conversation((await case.state()).values["messages"]))
    await case.ask("预算改成500元")
    current = (await case.state()).values["messages"]
    assert conversation(current)[:len(before)] == before
    assert len(catalogue(case)) == 1 and "body" not in catalogue(case)[0]
    assert case.requests[0]["messages"][0] == case.requests[1]["messages"][0]
    assert not any(m.name == "skill_catalog" for m in current)


async def test_edit_body_delete_last_skill_and_restart_append_without_rewriting(case):
    skill = case.personal.save("buyer", "背包方案", "通勤", "轻便优先")
    await case.ask("开始")
    before = deepcopy(conversation((await case.state()).values["messages"]))
    changed = case.personal.save("buyer", "背包方案", "通勤", "耐用优先",
        skill_id=skill["id"], expected_version="1")
    await case.ask("比较")
    assert catalogue(case)[0]["version"] == changed["version"] == "2"
    await case.container.orchestrator._sessions.invalidate("session")
    case.personal.delete("buyer", skill["id"], "2")
    await case.ask("还有方案吗")
    assert catalogue(case) == []
    assert conversation((await case.state()).values["messages"])[:len(before)] == before
    with pytest.raises(LookupError):
        case.personal.load("buyer", skill["id"], "1")


async def test_empty_catalog_and_noop_save_do_not_generate_versions(case):
    await case.ask("开始")
    await case.ask("继续")
    assert catalogue(case) == []
    one = case.personal.save("buyer", "背包", "通勤", "轻便")
    two = case.personal.save("buyer", " 背包 ", "通勤", "轻便", skill_id=one["id"], expected_version="1")
    assert one == two
    with pytest.raises(BuyerSkillConflict):
        case.personal.save("buyer", "背包", "通勤", "轻便", skill_id=one["id"], expected_version="0")


async def test_checkpoint_without_visible_catalog_reseeds_at_next_user_only(case):
    await case.ask("开始")
    session = case.container.orchestrator._sessions._agents["session"]
    claim = case.container.orchestrator._sessions._claims["session"]
    token = ShoppingContext.set(ShoppingContextSnapshot("session", "buyer", "zh-CN", "CNY",
                                                       session_fence=claim.fence))
    try:
        await session.graph.aupdate_state(session.config, {"messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES), AIMessage(content="已整理历史")]})
    finally:
        ShoppingContext.reset(token)
    await case.ask("继续")
    assert catalogue(case) == []


async def test_cancel_before_accepting_input_does_not_advance_watermark(case, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = case.personal.list
    def delayed(*args, **kwargs):
        entered.set()
        release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(case.personal, "list", delayed)
    task = asyncio.create_task(case.ask("开始"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not case.requests
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    monkeypatch.setattr(case.personal, "list", original)
    assert not (await case.ask("重试")).error
    assert catalogue(case) == []


async def test_cancel_after_accepting_input_commits_receipt_and_resume_does_not_duplicate(case, monkeypatch):
    original = type(case.model)._agenerate
    entered = asyncio.Event()
    async def cancelled(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(type(case.model), "_agenerate", cancelled)
    task = asyncio.create_task(case.ask("开始"))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not any(m.name == "skill_catalog" for m in (await case.state()).values["messages"])
    monkeypatch.setattr(type(case.model), "_agenerate", original)
    await case.ask("重试")
    assert catalogue(case) == []


def test_snapshot_ignores_order_and_timestamp_but_detects_effective_changes():
    a = {"id":"a","version":"1","content_hash":"hash-a","updated_at":"yesterday"}
    b = {"id":"b","version":"1","content_hash":"hash-b"}
    base = skill_directory([], [a, b])
    assert base == skill_directory([], [b, {**a,"updated_at":"today"}])
    for changed in ({**a,"content_hash":"new"}, {**a,"title":"新名字"},
                    {**a,"version":"2"}, {**a,"expires_at":"2100-01-01"}):
        assert base != skill_directory([], [changed,b])
    assert base != skill_directory([a], [b])


async def test_stale_selection_is_only_masked_in_request_and_current_requires_metadata(case):
    from langchain.agents.middleware import ModelRequest
    from app.application.runtime.skills import SkillReferenceMiddleware

    token=ShoppingContext.set(ShoppingContextSnapshot('projection','buyer','zh-CN','CNY'))
    try:
        ShoppingContext.set_capability_digest(case.registry.bind_session('projection','buyer'))
        old=HumanMessage(name='selected_skill_reference',content='旧私有正文',additional_kwargs={
            'skill_activation':{'id':'personal-deleted','version':'1','contentHash':'old'}})
        buyer=HumanMessage(name='buyer',content='新一轮')
        messages=[old,buyer]
        before=[m.model_dump_json() for m in messages]
        received=[]
        async def handler(request):
            received.append(request)
        middleware=SkillReferenceMiddleware(case.registry,case.personal,set())
        from app.application.runtime.context import RequestContextMiddleware
        policy=RequestContextMiddleware(None,None,case.container.settings,system_prompt="",tools=[],
                                        skill_source=middleware,summary_enabled=False)
        request=ModelRequest(model=case.model,messages=messages,tools=[],state={'messages':messages},runtime=None)
        projected,_,_=await policy.prepare(request)
        received.append(projected)
        assert '旧私有正文' not in received[0].messages[0].text
        assert [m.model_dump_json() for m in messages]==before
        current=HumanMessage(name='selected_skill_reference',content='没有可信版本的正文')
        projected,_,_=await policy.prepare(request.override(messages=[buyer,current]))
        received.append(projected)
        assert all('没有可信版本的正文' not in m.text for m in received[-1].messages)
    finally:
        ShoppingContext.reset(token)


async def test_buyer_isolation_and_watermark_owner_fail_closed(case):
    other = case.personal.save("other", "私有方案", "秘密用途", "秘密正文")
    await case.ask("开始")
    assert other["id"] not in str(catalogue(case))
    calls = len(case.requests)
    assert (await case.ask("继续", buyer="other")).error
    assert len(case.requests) == calls


def test_public_publish_is_hot_but_revoke_and_strategy_change_remain_blocked(tmp_path):
    registry = CapabilityRegistry(tmp_path / "capabilities.db")
    publish(registry, document())
    first = registry.bind_session("s","b",allow_skill_updates=True)
    publish(registry, document(version="2"))
    with pytest.raises(CapabilityVersionChanged):
        registry.bind_session("s","b")
    assert registry.bind_session("s","b",allow_skill_updates=True) != first
    with pytest.raises(CapabilityVersionChanged):
        registry.load_skill("backpack","1",available_tools={"product_search_tool"},require_current=True)
    assert registry.load_skill("backpack","2",available_tools={"product_search_tool"},require_current=True)
    registry.revoke("skill","backpack","1",actor="fixture",reason="硬撤销")
    with pytest.raises(CapabilityVersionChanged):
        registry.bind_session("s","b",allow_skill_updates=True)
    registry.bind_session("s-new","b",allow_skill_updates=True)
    publish(registry, document("strategy"))
    with pytest.raises(CapabilityVersionChanged):
        registry.bind_session("s-new","b",allow_skill_updates=True)


async def test_catalogue_does_not_preload_body_and_deletion_remains_authoritative(case):
    skill = case.personal.save("buyer", "背包", "通勤", "轻便优先")
    assert not (await case.ask("选购背包")).error
    assert "轻便优先" not in json.dumps(case.requests[-1],ensure_ascii=False)
    saved_request = deepcopy(case.requests[0])
    case.personal.delete("buyer", skill["id"], "1")
    assert not (await case.ask("普通选购")).error
    assert case.requests[0] == saved_request
    assert not any(m.get("name") == "selected_skill_reference" for m in case.requests[-1]["messages"])
    assert catalogue(case) == []


async def test_deleted_active_skill_is_rejected_before_model(case):
    skill = case.personal.save("buyer", "背包", "通勤", "轻便")
    case.personal.delete("buyer", skill["id"], "1")
    from app.application.agents.skill_access import read_skill
    token=ShoppingContext.set(ShoppingContextSnapshot("deleted","buyer","zh-CN","CNY"))
    try:
        with pytest.raises(LookupError):
            read_skill(case.registry,case.personal,set(),skill["id"],"1")
        assert not case.requests
    finally:ShoppingContext.reset(token)


@pytest.mark.parametrize("mode", ["legacy", "append_only"])
@pytest.mark.parametrize("scenario", ["unchanged", "edit", "delete"])
async def test_existing_harness_skill_replay_uses_native_orchestrator_and_sqlite(tmp_path, monkeypatch, mode, scenario):
    from scripts.eval.harness.cache_replay import run_skill_replay
    from app.infrastructure.throttle import GatewayThrottle
    from tests.test_retrieval import _settings
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        version = None if count >= 3 and scenario == "delete" else ("2" if count >= 3 and scenario == "edit" else "1")
        response = completion()
        response["choices"][0]["message"]["content"] = json.dumps({"available":version is not None,"version":version})
        return httpx.Response(200,json=response)
    model = await client_model(tmp_path,handler)
    monkeypatch.setattr("scripts.eval.harness.cache_replay.create_chat_model",lambda *a,**k:model)
    row = await run_skill_replay({"id":"skill-"+scenario,"scenario":scenario},
        replace(_settings(tmp_path),skill_catalog_mode=mode),GatewayThrottle(1,0),
        tmp_path/"replay",0,None,{"output_limit":128,"request_timeout_seconds":10})
    assert row["passed"] and len(row["usage"]) == 4 and len(row["transcript"]) == 4


async def test_native_approval_resume_defers_catalog_change_until_next_buyer_turn(tmp_path, monkeypatch):
    from app.application.runtime.events import approval_event
    model = ScriptedModel(mode="approval")
    container = await _container(tmp_path, monkeypatch, model)
    store = PreferenceStore()
    factory = container.orchestrator._sessions._main_factory
    factory._preference_store = store
    try:
        await container.orchestrator.handle_intent(SubmitIntentInput("s","buyer","zh-CN","CNY","记住轻便"))
        session = container.orchestrator._sessions._agents["s"]
        pending = approval_event((await session.graph.aget_state(session.config)).interrupts)
        factory.buyer_skill_store.save("buyer","新方案","用途","正文")
        result = await container.orchestrator.handle_intent(
            SubmitIntentInput("s","buyer","zh-CN","CNY","拒绝",confirmations=({
                "interrupt_id":pending.reply_id+":"+pending.tool_calls[0].id,"approved":False},)),
            )
        assert result.error is None and store.items == []
        # 审批只绑定原记忆操作；后续模型读取最新目录，不自动激活新增方案。
        assert "新方案" in next(m.content for m in model.seen[-1] if m.name == "skill_catalog")
    finally:
        await container.shutdown()


async def test_catalog_and_watermark_share_session_fencing_transaction(case):
    await case.ask("开始")
    sessions = case.container.orchestrator._sessions
    old = sessions._claims["session"]
    session = sessions._agents["session"]
    before = await session.graph.aget_state(session.config)
    await case.container.session_store.claim("session", buyer_id="buyer")
    token = ShoppingContext.set(ShoppingContextSnapshot("session","buyer","zh-CN","CNY",session_fence=old.fence))
    try:
        with pytest.raises(StaleSessionWrite):
            await session.graph.aupdate_state(session.config, {"messages":[HumanMessage(content="迟到目录")]})
    finally:
        ShoppingContext.reset(token)
    assert (await session.graph.aget_state(session.config)).values == before.values


def test_skill_evaluation_scorer_rejects_stale_versions_and_string_booleans():
    from scripts.eval.harness.cache_replay import check_skill_answer
    assert all(check_skill_answer('{"available":false,"version":null}',None).values())
    for text in ('{"available":true,"version":"1"}','{"available":"false","version":null}','已删除'):
        assert not all(check_skill_answer(text,None).values())


async def test_explicit_compaction_can_archive_old_skill_but_pins_current_catalog_and_activation(tmp_path):
    from tests.test_langgraph_context import policy
    messages = []
    for i in range(6):
        messages += [HumanMessage(content="选购",name="buyer",id=f"u{i}"),
                     AIMessage(content="回答",id=f"a{i}")]
    before = deepcopy(messages)
    references={"demo":{"id":"demo","version":"1","content_hash":"a"*64}}
    token = ShoppingContext.set(ShoppingContextSnapshot("s","buyer","zh-CN","CNY"))
    try:
        state={"messages":messages,"read_tool_messages":[],"loaded_skills":references,"skill_turn_id":"u5"}
        update = await policy(tmp_path).compact_checkpoint(state,force=True)
        assert {**state,**update}["loaded_skills"]==references
        assert messages == before
    finally:
        ShoppingContext.reset(token)
