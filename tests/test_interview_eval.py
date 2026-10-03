"""面试评测本身必须识别错误结果，不能因为缺轨迹或固定答案而假绿。"""
import json
from pathlib import Path
import pytest
import yaml
from tests.trade_test_helpers import confirmation_env
from scripts.eval.evidence import evaluate_trace_assertions
from scripts.eval_regression import build_judge_rubric
from scripts.eval.http_actions import validate_http_actions


def evaluate(kind, events, **fields):
    return evaluate_trace_assertions([{"kind":kind,"criterion":"检查",**fields}],events)[0]["pass"]


def presentation(total=41800):
    return {"type":"recommendation.result","payload":{"hits":[{"default_sku_id":"P1001-S1","quantity":2,
        "currency":"CNY","landed_price":{"total_amount_minor":total}}]}}


def test_eight_cases_use_existing_scoring_without_judge():
    cases = yaml.safe_load(Path("eval/interview.yaml").read_text())["cases"]
    assert len(cases) == 8 and len({c["id"] for c in cases}) == 8
    for case in cases:
        assert not any(build_judge_rubric(case["rubric"],case["deterministic"]).values())
        validate_http_actions(case)


def test_wrong_amount_and_absent_presentation_fail():
    expected={"event":"recommendation.result","items":[{"sku_id":"P1001-S1","quantity":2,"total_amount_minor":41800}]}
    assert evaluate("presentation_matches",[presentation()],**expected)
    assert not evaluate("presentation_matches",[presentation(37800)],**expected)
    assert not evaluate("presentation_matches",[],**expected)


def test_terminal_needs_model_evidence_and_rejects_extra_summary():
    request={"type":"eval.model_request","payload":{}}
    assert evaluate("presentation_terminal",[request,presentation()],event="recommendation.result")
    assert not evaluate("presentation_terminal",[presentation()],event="recommendation.result")
    assert not evaluate("presentation_terminal",[request,presentation(),request],event="recommendation.result")


def test_turn_filter_cannot_borrow_prior_success():
    events=[presentation(),{"type":"eval.turn.complete","payload":{"turn_index":1}},
        presentation(37800),{"type":"eval.turn.complete","payload":{"turn_index":2}}]
    assert not evaluate("presentation_matches",events,turn=2,event="recommendation.result",
        items=[{"sku_id":"P1001-S1","total_amount_minor":41800}])
    assert not evaluate("presentation_matches",events,turn=3,event="recommendation.result",items=[])


def test_recovery_requires_failure_then_success():
    def tool(status): return {"type":"eval.tool_result","payload":{"tool":"search","state":status}}
    rule={"failure_tools":["search"],"success_event":"recommendation.result"}
    assert evaluate("business_recovered",[tool("error"),presentation()],**rule)
    assert not evaluate("business_recovered",[presentation(),tool("error")],**rule)


def test_required_tool_counts_observed_native_execution():
    events=[{"type":"eval.tool_result","payload":{"tool":"conversation_fact_lookup","state":"success"}}]
    assert evaluate("required_tools",events,tools=["conversation_fact_lookup"])
    assert not evaluate("forbidden_tools",events,tools=["conversation_fact_lookup"])


def test_pending_case_needs_real_confirmation_not_just_no_order():
    expected={"total_amount_minor":15400}
    assert not evaluate("confirmation_prepared",[],expected_payload=expected)
    events=[{"type":"eval.confirmations.snapshot","payload":{"confirmations":[{
        "action":"create","status":"pending","result":None,"payload":expected}]}}]
    assert evaluate("confirmation_prepared",events,expected_payload=expected)


def test_completed_delegates_do_not_make_an_empty_main_reply_pass():
    events=[{"type":"final.result","payload":{"text":"", "status":"completed", "stop_reason":None}}, {"type":"eval.turn.complete","payload":{"turn_index":1}}]
    assert not evaluate("answer_present",events)
    events[0]["payload"]["text"]="结果已交付"
    assert evaluate("answer_present",events)


def test_two_deliveries_of_same_category_do_not_cover_two_requested_categories():
    events=[{"type":"eval.tool_result","payload":{"tool":"product_search_tool","result":[
        {"hits":[{"product_id":"bag","category":"旅行装备"},{"product_id":"ear","category":"数码配件"}]}]}}]
    for _ in range(2):
        events.append({"type":"eval.tool_result","payload":{"tool":"task_dispatch","result":[
            {"status":"completed","candidates":[{"product_id":"bag"}],"evidence_refs":["ref"]}]}})
    rule={"minimum":2,"categories":["旅行装备","数码配件"]}
    assert not evaluate("delegation_delivered",events,**rule)
    events[-1]['payload']['result'][0]['candidates'][0]['product_id']='ear'
    assert evaluate("delegation_delivered",events,**rule)


@pytest.mark.asyncio
async def test_isolated_production_api_fixture(tmp_path):
    from dataclasses import replace
    from tests.test_retrieval import _settings
    from tests.test_langgraph_runtime import ScriptedModel
    from scripts.eval.interview_runtime import isolated_runtime, LocalCollector
    from scripts import eval_regression as runner
    from unittest.mock import patch
    import httpx
    model=ScriptedModel(mode="search")
    case=yaml.safe_load(Path("eval/interview.yaml").read_text())["cases"][0]
    settings = replace(_settings(tmp_path), llm_api_key="test-key", embedding_api_key="test-key")
    async with isolated_runtime(tmp_path/"runtime", settings, model_factory=lambda *a,**kw:model) as (app,container):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            with patch.object(runner,"SessionEventCollector",LocalCollector):
                result=await runner.run_case(client,case,"",identity_policy=app.state.identity_policy)
        assert result["verdict"] == "PASS"
        from app.application.runtime.execution import graph_step_limit
        for session in container.orchestrator._sessions._agents.values():
            assert session.config['recursion_limit'] == graph_step_limit(session.graph, settings.agent_max_model_rounds)
        assert any(e["type"] == "tool.result" for e in result["trace_events"])
        assert container.settings.data_dir == tmp_path/"runtime"


@pytest.mark.asyncio
async def test_compaction_uses_http_revision_and_records_actual_completion():
    import httpx
    from scripts.eval.http_actions import compact_context
    calls=[]
    def respond(request):
        calls.append(request)
        if request.method == "POST":
            assert json.loads(request.content)["expected_revision"] == 7
            return httpx.Response(202,json={"operation_id":"operation","status":"running"})
        if request.url.path.endswith("/context"):
            return httpx.Response(200,json={"revision":7})
        return httpx.Response(200,json={"operation_id":"operation","status":"completed"})
    events=[]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await compact_context(client,"http://test","buyer","session",events)
    assert evaluate("context_compacted",events)
    assert not evaluate("context_compacted",[{"type":"eval.context.operation","payload":{"status":"noop"}}])


@pytest.mark.asyncio
async def test_replay_verifies_actual_inventory_and_order_count(confirmation_env):
    import httpx
    from types import SimpleNamespace
    from unittest.mock import patch
    from scripts.eval_interview import replay_confirmations
    from scripts import eval_regression as runner
    from tests.test_eval_http_actions import _api, _action, address
    from app.application.usecases.order_usecases import OrderItemInput
    from scripts.eval.http_actions import execute_confirmation_action
    env=confirmation_env
    prepared=await env.service.prepare_order("buyer","session",[OrderItemInput("P1001","P1001-S1",1)],address())
    events=[]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_api(env))) as client:
        await execute_confirmation_action(client,"http://test","buyer","session",_action(),[prepared["confirmation"]],events)
        with patch.object(runner,"BASE_URL","http://test"):
            await replay_confirmations(client,SimpleNamespace(trade_store=env.store),{"trace_events":events})
    result=events[-1]["payload"]
    assert result["orders_before"] == result["orders_after"] == 1
    assert result["inventory_before"] == result["inventory_after"] == {"P1001-S1":49}


@pytest.mark.asyncio
async def test_consecutive_real_factories_own_distinct_http_clients(tmp_path):
    from dataclasses import replace
    from tests.test_retrieval import _settings
    from scripts.eval.interview_runtime import isolated_runtime
    settings=replace(_settings(tmp_path),llm_api_key="test-key",embedding_api_key="test-key",llm_base_url="http://127.0.0.1:9/v1")
    clients=[]
    for index in range(2):
        async with isolated_runtime(tmp_path/str(index),settings) as (_,container):
            session=container.orchestrator._sessions._main_factory.build()
            client=session.context_policy.model.client
            assert client is not None and not client.is_closed()
            assert all(client is not previous for previous in clients)
            clients.append(client)
        assert client.is_closed()
