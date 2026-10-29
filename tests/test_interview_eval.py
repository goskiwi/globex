"""面试评测本身必须识别错误结果，不能因为缺轨迹或固定答案而假绿。"""
import json
from pathlib import Path
import pytest
import yaml
from tests.native_trace_helpers import tool_frames, final_frames, presentation_frame
from scripts.eval.trace import tool_observations
from tests.trade_test_helpers import confirmation_env
from scripts.eval.evidence import evaluate_trace_assertions
from scripts.eval_regression import build_judge_rubric
from scripts.eval.http_actions import validate_http_actions


def evaluate(kind, events, **fields):
    return evaluate_trace_assertions([{"kind":kind,"criterion":"检查",**fields}],events)[0]["pass"]


def presentation(total=41800):
    return presentation_frame('recommendation', {'hits':[{'default_sku_id':'P1001-S1','quantity':2,
        'currency':'CNY','landed_price':{'total_amount_minor':total}}]})


def test_eight_cases_use_existing_scoring_without_judge():
    cases = yaml.safe_load(Path("eval/interview.yaml").read_text())["cases"]
    assert len(cases) == 8 and len({c["id"] for c in cases}) == 8
    for case in cases:
        assert not any(build_judge_rubric(case["rubric"],case["deterministic"]).values())
        validate_http_actions(case)


def test_wrong_amount_and_absent_presentation_fail():
    expected={"presentation_kind":"recommendation","items":[{"sku_id":"P1001-S1","quantity":2,"total_amount_minor":41800}]}
    assert evaluate("presentation_matches",[presentation()],**expected)
    assert not evaluate("presentation_matches",[presentation(37800)],**expected)
    assert not evaluate("presentation_matches",[],**expected)


def test_terminal_needs_model_evidence_and_rejects_extra_summary():
    request={'type':'STEP_STARTED','stepName':'model:sample'}
    assert evaluate("presentation_terminal",[request,presentation()],presentation_kind="recommendation")
    assert not evaluate("presentation_terminal",[presentation()],presentation_kind="recommendation")
    assert not evaluate("presentation_terminal",[request,presentation(),request],presentation_kind="recommendation")


def test_turn_filter_cannot_borrow_prior_success():
    events=[presentation(),{"type":"eval.turn.complete","payload":{"turn_index":1}},
        presentation(37800),{"type":"eval.turn.complete","payload":{"turn_index":2}}]
    assert not evaluate("presentation_matches",events,turn=2,presentation_kind="recommendation",
        items=[{"sku_id":"P1001-S1","total_amount_minor":41800}])
    assert not evaluate("presentation_matches",events,turn=3,presentation_kind="recommendation",items=[])


def test_recovery_requires_failure_then_success():
    def tool(status): return tool_frames('search',status=status)
    rule={"failure_tools":["search"],"presentation_kind":"recommendation"}
    assert evaluate("business_recovered",[*tool("error"),presentation()],**rule)
    assert not evaluate("business_recovered",[presentation(),*tool("error")],**rule)


def test_required_tool_counts_observed_native_execution():
    events=tool_frames('conversation_fact_lookup')
    assert evaluate("required_tools",events,tools=["conversation_fact_lookup"])
    assert not evaluate("forbidden_tools",events,tools=["conversation_fact_lookup"])


def test_pending_case_needs_real_confirmation_not_just_no_order():
    expected={"total_amount_minor":15400}
    assert not evaluate("confirmation_prepared",[],expected_payload=expected)
    events=[{"type":"eval.confirmations.snapshot","payload":{"confirmations":[{
        "action":"create","status":"pending","result":None,"payload":expected}]}}]
    assert evaluate("confirmation_prepared",events,expected_payload=expected)


def test_completed_delegates_do_not_make_an_empty_main_reply_pass():
    events=[*final_frames(),{'type':'eval.turn.complete','payload':{'turn_index':1}}]
    assert not evaluate("answer_present",events)
    events[0]['messages'][0]['content']='结果已交付'
    assert evaluate("answer_present",events)


def research_candidate(pid,category):
    return {'product_id':pid,'sku_id':pid+'-S1','facts':{'product':{'product_id':pid,'category':category},'sku':{'sku_id':pid+'-S1'}}}


def research_events(candidates):
    events=[]
    for index,candidate in enumerate(candidates):
        task_id='task-'+str(index)
        events.extend(tool_frames('task_dispatch',{'status':'completed','candidates':[candidate],'evidence_refs':['ctx_actual'],'scope':{'task_id':task_id}}))
    reports=[{'task_id':'task-'+str(i),'items':[c]} for i,c in enumerate(candidates)]
    events.append(presentation_frame('research',{'task_ids':[r['task_id'] for r in reports],'reports':reports}))
    return events


def test_two_deliveries_of_same_category_do_not_cover_two_requested_categories():
    bag=research_candidate('bag','旅行装备');ear=research_candidate('ear','数码配件')
    rule={'minimum':2,'categories':['旅行装备','数码配件']}
    assert not evaluate('delegation_delivered',research_events([bag,bag]),**rule)
    assert evaluate('delegation_delivered',research_events([bag,ear]),**rule)


@pytest.mark.parametrize('failure',['missing_sku','mismatched_sku','missing_provenance','no_report','invented_presented_sku'])
def test_original_research_requires_actual_report_with_verified_skus(failure):
    from copy import deepcopy
    events=research_events([research_candidate('bag','旅行装备'),research_candidate('ear','数码配件')])
    receipt=json.loads(events[3]['content'])
    if failure=='missing_sku':receipt['data']['candidates'][0].pop('sku_id')
    elif failure=='mismatched_sku':receipt['data']['candidates'][0]['facts']['sku']['sku_id']='another-sku'
    elif failure=='missing_provenance':receipt['data']['evidence_refs']=[]
    elif failure=='no_report':events.pop()
    else:
        events=deepcopy(events)
        events[-1]['snapshot']['presentations'][0]['payload']['reports'][0]['items']=[research_candidate('invented','旅行装备')]
    events[3]['content']=json.dumps(receipt)
    assert not evaluate('delegation_delivered',events,minimum=2,categories=['旅行装备','数码配件'])



@pytest.mark.asyncio
async def test_isolated_production_api_fixture(tmp_path):
    from dataclasses import replace
    from tests.test_retrieval import _settings
    from tests.test_langgraph_runtime import ScriptedModel
    from scripts.eval.agent_runtime import isolated_runtime, capture_evaluation
    from scripts import eval_regression as runner
    from unittest.mock import patch
    import httpx
    model=ScriptedModel(mode="search")
    case=yaml.safe_load(Path("eval/interview.yaml").read_text())["cases"][0]
    settings = replace(_settings(tmp_path), llm_api_key="test-key", embedding_api_key="test-key")
    async with isolated_runtime(tmp_path/"runtime", settings, model_factory=lambda *a,**kw:model) as (app,container):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            events=[]
            with capture_evaluation(events):
                result=await runner.run_case(client,case,"",identity_policy=app.state.identity_policy,evidence_events=events)
        assert result["verdict"] == "PASS"
        from app.application.runtime.execution import graph_step_limit
        for session in container.orchestrator._sessions._agents.values():
            assert session.config['recursion_limit'] == graph_step_limit(session.graph, settings.agent_max_model_rounds)
        assert any(e['type']=='TOOL_CALL_RESULT' for e in result['trace_events'])
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
    from scripts.eval.agent_runtime import isolated_runtime
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
