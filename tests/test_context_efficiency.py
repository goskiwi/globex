"""原生状态增量、请求投影与分页证据，不维护旧 Agent 执行器。"""
from copy import deepcopy
from dataclasses import replace
import json
import httpx
import pytest
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from app.infrastructure.context_usage import context_diagnostic_sink
from app.infrastructure.context import ShoppingContext,ShoppingContextSnapshot
from app.application.runtime.state_projection import *
from app.application.runtime.projections import share_identical_products,read_output,EVIDENCE_RULES
from app.infrastructure.context_products import token_estimate
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from tests.native_context_helpers import history,policy,apply
from tests.shopping_state_helpers import work_fixture

@pytest.fixture(autouse=True)
def scope():
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)

def test_delta_constraint_removal_and_broken_chain_restore_are_explicit():
    work = {'goal': '买包', 'filters': {str(i): {'source': '防水轻便' * 30, 'message_id': 'm1'} for i in range(8)},
            'selected': ['P1-S1'], 'excluded': [], 'source_message_id': 'm1', 'latest_request': '原始请求'}
    history = [snapshot_message(work)]
    changed = deepcopy(work)
    del changed['filters']['3']
    changed['filters']['2'] = {'source': '允许较重', 'message_id': 'm2'}
    hint = next_state_message(history, changed)
    assert hint.name == 'shopping_state_delta'
    history.append(hint)
    assert semantic_state(replay_state(history)[0]) == semantic_state(projected_state(changed))
    assert replay_state(history[1:])[0] is None
    repaired = project_state_messages(history[1:], changed)
    assert len(repaired) == 1 and replay_state(repaired)[0]['filters']['2']['source'] == '允许较重'
    assert 'latest_request' not in repaired[0].text
    # 没有语义变化，只有审计来源的新消息ID，不制造新模型状态。
    changed['filters']['2']['message_id'] = 'm3'
    assert next_state_message(history, changed) is None

def test_snapshot_rebases_after_bounded_number_of_deltas():
    work = {'goal': '买包', 'filters': {str(i): {'source': '材料限制' * 60} for i in range(8)}}
    history = [snapshot_message(work)]
    for i in range(9):
        work['filters']['budget'] = {'value': str(100 + i)}
        history.append(next_state_message(history, work))
    assert history[-1].name == 'shopping_state'
    assert replay_state(history)[2] == 0

def test_diagnostic_html_keeps_unknowns_and_escapes_events(tmp_path):
    from tests.test_harness_evaluation import manifest, rows
    from scripts.eval.harness.report import render
    data = rows()
    data[0]['context_diagnostics'] = [{'type': 'lookup', 'status': 'error', 'error_code': '<script>bad</script>'}]
    report = render(tmp_path, manifest(), data)
    assert report['strategies']['current']['efficiency_diagnostics']['lookup_errors'] == 1
    html = (tmp_path/'report.html').read_text()
    assert '调用增量与治理归因' in html and '&lt;script&gt;bad' in html
    assert '<script>bad</script>' not in html

@pytest.mark.parametrize('mode,success', [('strict', False), ('bounded', True)])
async def test_lookup_repairs_only_unambiguous_parameters_and_keeps_all_pages(tmp_path, mode, success):
    store = ContextEvidenceStore(tmp_path / 'e.db')
    hits = [{'product_id': f'P{i}', 'title': '背包', 'skus': [{'sku_id': f'P{i}-S1', 'spec': '黑色', 'price_major': 10 + i, 'currency': 'CNY'}]} for i in range(7)]
    ref = await store.save('b', 's', 'display_batch', {'hits': hits})
    lookup = build_conversation_fact_lookup(store, mode=mode)
    events = []; token = context_diagnostic_sink.set(events.append)
    try:
        result = await lookup(result_ref=ref, fields='skus.sku_id,skus.spec,skus.price_major,skus.currency', limit=10)
        assert (result.state.value == 'success') == success
        if not success:
            assert events[-1]['error_code'] == 'invalid_request'
            return
        payload = json.loads(result.text)
        page = payload['records'][0]['data']
        assert page['hits'] == hits[:5] and page['next_offset'] == 5 and page['total'] == 7
        assert not payload['page_contract']['complete_selection']
        assert events[-1]['limit_capped'] and events[-1]['fields_normalized']
        more = json.loads((await lookup(result_ref=ref, fields='skus', offset=5)).text)
        assert more['records'][0]['data']['hits'] == hits[5:]
        assert more['page_contract']['complete_selection']
        assert token_estimate(payload) <= 3000 and token_estimate(more) <= 3000
    finally:
        context_diagnostic_sink.reset(token)

async def test_lookup_rejects_unknown_fields_and_cross_buyer_reference(tmp_path):
    store = ContextEvidenceStore(tmp_path / 'e.db')
    ref = await store.save('other', 's', 'display_batch', {'hits': [{'product_id': 'P99'}]})
    lookup = build_conversation_fact_lookup(store, mode='bounded')
    assert (await lookup(fields='skus.secret')).state.value == 'error'
    assert (await lookup(limit=0)).state.value == 'error'
    result = json.loads((await lookup(result_ref=ref)).text)
    assert result['records'] == [] and not result['page_contract']['complete_selection']

async def test_synthetic_lookup_trace_preserves_actual_parameters_pages_and_errors(tmp_path):
    from app.infrastructure.context_usage import evaluation_evidence_sink
    store = ContextEvidenceStore(tmp_path/'e.db')
    ref = await store.save('b', 's', 'display_batch', {'hits': [
        {'product_id': f'P{i}', 'skus': [{'sku_id': f'P{i}-S1', 'spec': '黑色', 'price_major': i, 'currency': 'CNY'}]}
        for i in range(7)]})
    lookup = build_conversation_fact_lookup(store, mode='bounded')
    events = []; token = evaluation_evidence_sink.set(events.append)
    try:
        response = await lookup(result_ref=ref, fields='skus.price_major,skus.currency', limit=10)
        page = events[-1]
        assert page['kind'] == 'lookup_result'
        assert page['payload']['arguments']['limit'] == 10
        assert page['payload']['effective_limit'] == 5
        assert page['payload']['result'] == json.loads(response.text)
        assert page['payload']['result']['records'][0]['data']['next_offset'] == 5
        await lookup(result_ref=ref, fields='unknown-secret-field')
        assert events[-1]['payload']['state'] == 'error'
        assert events[-1]['payload']['arguments']['fields'] == 'unknown-secret-field'
        assert events[-1]['payload']['error_code'] == 'invalid_fields'
    finally:
        evaluation_evidence_sink.reset(token)

async def test_bounded_page_budget_includes_evidence_envelope(tmp_path):
    store = ContextEvidenceStore(tmp_path / 'e.db')
    hits = [{'product_id': f'P{i}', 'description': '限制' * 500, 'title': '包'} for i in range(6)]
    ref = await store.save('b', 's', 'display_batch', {'hits': hits, 'query_conditions': {'query': '旅行' * 70}})
    lookup = build_conversation_fact_lookup(store, mode='bounded')
    collected = []; offset = 0
    while True:
        chunk = await lookup(result_ref=ref, offset=offset)
        assert chunk.state.value == 'success'
        payload = json.loads(chunk.text)
        assert token_estimate(payload) <= 3000
        page = payload['records'][0]['data']; collected.extend(page['hits'])
        if page['next_offset'] is None: break
        assert page['next_offset'] > offset
        offset = page['next_offset']
    assert collected == hits

def test_efficiency_suite_preserves_model_permissions_and_default_switches(tmp_path):
    from scripts.eval.harness.contracts import load_suite, ROOT
    from tests.test_retrieval import _settings
    suite = load_suite(ROOT / 'eval/harness/v1/efficiency-suite.json')
    assert suite['profiles']['efficiency']['repetitions'] == 3
    assert len(suite['profiles']['long_efficiency']['cases']) == 2
    settings = _settings(tmp_path)
    assert settings.context_state_mode == 'snapshot' and settings.context_lookup_mode == 'strict'
    assert settings.context_prune_low_ratio == 1.0
    assert suite['strategies']['candidate']['overrides']['context_prune_low_ratio'] == .65

async def test_delta_graph_does_not_accumulate_unchanged_state_and_restores(tmp_path):
    from langchain.agents import create_agent
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from app.application.runtime.working_state import WorkingStateMiddleware
    from tests.native_model_helpers import client_model,completion
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content));return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    config={'configurable':{'thread_id':'s'}}
    try:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'graph.db')) as saver:
            graph=create_agent(model,system_prompt='固定规则',middleware=[
                WorkingStateMiddleware('delta'),policy(tmp_path,system_prompt='固定规则',working_state_mode='delta')],checkpointer=saver)
            for text in ['预算300元，选中P1003-S1，排除P1002。','还有别的款吗？','预算是多少？']:
                await graph.ainvoke({'shopping_work':work_fixture(budget=300),'messages':[HumanMessage(name='b',content=text)]},config)
            state=await graph.aget_state(config)
            assert len([m for m in state.values['messages'] if m.name in STATE_NAMES])==1
            await graph.ainvoke({'shopping_work':work_fixture(budget=180),'messages':[HumanMessage(name='b',content='预算改为180元')]},config)
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'graph.db')) as saver:
            graph=create_agent(model,system_prompt='固定规则',middleware=[
                WorkingStateMiddleware('delta'),policy(tmp_path,system_prompt='固定规则',working_state_mode='delta')],checkpointer=saver)
            state=await graph.aget_state(config)
            assert replay_state(state.values['messages'])[0]['filters']['price_max_major']==180
            await graph.ainvoke({'shopping_work':work_fixture(),'messages':[HumanMessage(name='b',content='换个需求，现在想买耳机。')]},config)
            state=await graph.aget_state(config)
            assert state.values['shopping_work']['filters']['price_max_major'] is None
            assert replay_state(state.values['messages'])[0]['filters']['price_max_major'] is None
        assert all(r['messages'][0]==requests[0]['messages'][0] for r in requests)
    finally:await model.aclose()

def test_projection_anchor_and_compact_copy_are_lossless():
    user=HumanMessage(name='b',content='新预算',id='u')
    work={'filters':{},'source_message_id':'u'}
    assert project_state_messages([user],work)[0]==user
    state=history(4);before=deepcopy(state)
    compact=share_identical_products(state['messages'],compact_rules=True)
    assert state==before
    assert [read_output(m) for m in compact if isinstance(m,ToolMessage)]==[read_output(m) for m in state['messages'] if isinstance(m,ToolMessage)]
    assert sum(len(m.content) for m in compact if isinstance(m,ToolMessage))<=sum(len(m.content) for m in state['messages'] if isinstance(m,ToolMessage))

async def test_low_watermark_and_invalid_configuration(tmp_path):
    state=history(10)
    middleware=policy(tmp_path,product_tokens=1400,context_prune_low_ratio=.65)
    updates=await middleware.compact_checkpoint(state)
    assert updates['messages'] and updates['context_product_trigger']>=1400
    state=apply(state,updates)
    assert not (await middleware.compact_checkpoint(state))['messages']
    for ratio in [0,1.1]:
        with pytest.raises(ValueError):policy(tmp_path,context_prune_low_ratio=ratio)
    from app.application.runtime.working_state import WorkingStateMiddleware
    with pytest.raises(ValueError):WorkingStateMiddleware('bad')
