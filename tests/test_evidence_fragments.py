"""实际历史回查入口：同一不可变观察的字段页组成证据，不跨来源拼接。"""
from copy import deepcopy

import pytest
from langchain_core.messages import ToolMessage

from app.application.agents.handoff import ResearchSubmission
from app.application.runtime.handoff import TaskEvidence
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from app.application.usecases.research_report import project_research_report
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from tests.test_agent_handoff import factories, call, submission, task


@pytest.fixture(autouse=True)
def scope():
    token = ShoppingContext.set(ShoppingContextSnapshot('handoff', 'buyer', 'zh-CN', 'CNY'))
    try:
        yield
    finally:
        ShoppingContext.reset(token)


def product(**changes):
    return {'product_id': 'P1003', 'title': '合成背包', 'description': '目录原文：防泼水；电脑仓未知。',
            'weight_kg': .38, 'dimensions_cm': {'length': 50, 'width': 32, 'height': 24},
            'material_tags': ['合成聚合物'],
            'rating_summary': {'average': 4.2, 'review_count': 52},
            'skus': [{'sku_id': 'P1003-S1', 'spec': '黑色', 'price_major': 129, 'currency': 'CNY', 'stock': 80},
                     {'sku_id': 'P1003-S2', 'spec': '蓝色', 'price_major': 149, 'currency': 'CNY', 'stock': 60}],
            **changes}


async def saved(store, card, when='2026-10-22T00:00:00Z'):
    return await store.save('buyer', 'handoff', 'products', {'observed_at': when, 'hits': [card]})


async def lookup_into(store, evidence, ref, fields, *, sku='P1003-S1', identifier='lookup'):
    result = await build_conversation_fact_lookup(store)(result_ref=ref, sku_id=sku, fields=fields)
    assert result.ok, result.text
    evidence.capture_lookup([ToolMessage(content='', name='conversation_fact_lookup', tool_call_id=identifier,
                                        artifact={'data': result.data})])
    return result.data


def report(evidence, *, skus=('P1003-S1',), fields=('description', 'weight_kg')):
    selected = ResearchSubmission(candidates=[{'product_id': 'P1003', 'sku_id': sku,
                                              'evidence_fields': list(fields)} for sku in skus])
    assert not evidence.validate(selected)
    result = evidence.delivered(selected).model_dump()
    result.update(scope={'task_id': 'facts', 'effective_search': {'parameters': {'target_currency': 'CNY', 'ship_to': 'CN'}}},
                  transaction_state='none')
    return project_research_report(result, ExchangeRateTable(), title='合成证据报告')


@pytest.mark.parametrize('fields_order', [('all', 'price'), ('price', 'all'), ('specs', 'identity', 'price')])
async def test_same_observation_field_pages_keep_read_values_and_source(tmp_path, fields_order):
    store = ContextEvidenceStore(tmp_path / 'evidence.db')
    ref = await saved(store, product())
    original = await store.get('buyer', 'handoff', ref)
    evidence = TaskEvidence('handoff')
    for index, fields in enumerate(fields_order):
        await lookup_into(store, evidence, ref, fields, identifier=f'page-{index}')
    item = report(evidence)['items'][0]
    assert item['missing_fields'] == []
    assert {e['field']: e['value'] for e in item['evidence']} == {
        'description': product()['description'], 'weight_kg': .38}
    assert item['facts']['result_ref'] == ref and item['facts']['historical'] is True
    assert item['facts']['sku']['price_major'] == 129 and item['facts']['sku']['stock'] == 80
    assert await store.get('buyer', 'handoff', ref) == original


async def test_same_observation_product_fields_are_shared_without_crossing_sku_prices(tmp_path):
    store = ContextEvidenceStore(tmp_path / 'evidence.db'); ref = await saved(store, product())
    evidence = TaskEvidence('handoff')
    await lookup_into(store, evidence, ref, 'all', sku='P1003-S1', identifier='all-black')
    await lookup_into(store, evidence, ref, 'price', sku='P1003-S2', identifier='price-blue')
    items = report(evidence, skus=('P1003-S1', 'P1003-S2'))['items']
    assert all(not item['missing_fields'] for item in items)
    assert [item['facts']['sku']['price_major'] for item in items] == [129, 149]
    assert [item['facts']['sku']['stock'] for item in items] == [80, 60]


async def test_interleaved_sources_preserve_their_own_fragments_without_mixing(tmp_path):
    store = ContextEvidenceStore(tmp_path / 'evidence.db')
    first = await saved(store, product(), '2026-09-01T00:00:00Z')
    second = await saved(store, product(description='第二份观察', weight_kg=.9))
    evidence = TaskEvidence('handoff')
    await lookup_into(store, evidence, first, 'all', identifier='first-all')
    await lookup_into(store, evidence, second, 'price', identifier='second-price')
    narrow = report(evidence)['items'][0]
    assert narrow['facts']['result_ref'] == second
    assert {m['field'] for m in narrow['missing_fields']} == {'description', 'weight_kg'}
    await lookup_into(store, evidence, first, 'price', identifier='first-price')
    restored = report(evidence)['items'][0]
    assert restored['facts']['result_ref'] == first and not restored['missing_fields']
    assert restored['facts']['product']['weight_kg'] == .38
    await lookup_into(store, evidence, second, 'all', identifier='second-all')
    current = report(evidence)['items'][0]
    assert current['facts']['result_ref'] == second
    assert current['facts']['product']['weight_kg'] == .9
    assert current['facts']['product']['description'] == '第二份观察'


@pytest.mark.parametrize('null_value', [None, ''])
async def test_explicit_null_and_empty_fields_remain_distinct_from_unread_fields(tmp_path, null_value):
    store = ContextEvidenceStore(tmp_path / 'evidence.db')
    card = product(description=null_value, weight_kg=None, dimensions_cm={}, material_tags=[],
                   rating_summary={'average': 0, 'review_count': 0})
    ref = await saved(store, card); evidence = TaskEvidence('handoff')
    await lookup_into(store, evidence, ref, 'all', identifier='all')
    await lookup_into(store, evidence, ref, 'price', identifier='price')
    item = report(evidence, fields=('description', 'weight_kg', 'dimensions_cm', 'material_tags', 'rating_summary'))['items'][0]
    assert item['facts']['product']['description'] == null_value
    assert item['facts']['product']['dimensions_cm'] == {} and item['facts']['product']['material_tags'] == []
    assert {m['field'] for m in item['missing_fields']} == {'description', 'weight_kg', 'dimensions_cm', 'material_tags'}
    assert item['evidence'] == [{'field': 'rating_summary', 'label': '目录评分与评价数', 'value': {'average': 0, 'review_count': 0}}]


def capture_current(evidence, card, ref=None):
    with observe_run_events(evidence.capture):
        TradeEventBus().publish('handoff', 'tool.result', {'tool': 'product_search_tool', 'result_ref': ref,
            'observed_at': '2026-10-22T00:00:00Z', 'hits': [card]})


@pytest.mark.parametrize('historical_first', [False, True])
async def test_current_source_remains_authoritative_over_other_historical_source(tmp_path, historical_first):
    store = ContextEvidenceStore(tmp_path / 'evidence.db')
    old = await saved(store, product(weight_kg=.9), '2026-09-01T00:00:00Z')
    fresh = product(description='当前已读资料', weight_kg=.4)
    current = await saved(store, fresh); evidence = TaskEvidence('handoff')
    if historical_first:
        await lookup_into(store, evidence, old, 'all', identifier='old-first')
    capture_current(evidence, fresh, current)
    await lookup_into(store, evidence, old, 'all', identifier='old-last')
    item = report(evidence)['items'][0]
    assert item['facts']['result_ref'] == current and item['facts']['historical'] is False
    assert item['facts']['product']['weight_kg'] == .4
    assert evidence.qualification[('P1003', 'P1003-S1')]['verified']


async def test_same_source_current_read_does_not_upgrade_an_unread_sku(tmp_path):
    store = ContextEvidenceStore(tmp_path / 'evidence.db'); card = product(); ref = await saved(store, card)
    evidence = TaskEvidence('handoff')
    await lookup_into(store, evidence, ref, 'all', sku='P1003-S2', identifier='historical-blue')
    capture_current(evidence, {**card, 'skus': [card['skus'][0]]}, ref)
    items = report(evidence, skus=('P1003-S1', 'P1003-S2'))['items']
    assert [item['facts']['historical'] for item in items] == [False, True]
    assert evidence.qualification[('P1003', 'P1003-S1')]['verified']
    assert not evidence.qualification[('P1003', 'P1003-S2')]['verified']


def test_no_source_reference_never_combines_unrelated_observations():
    evidence = TaskEvidence('handoff'); first = product()
    capture_current(evidence, first)
    capture_current(evidence, {'product_id': 'P1003', 'title': '第二次观察', 'skus': first['skus']})
    item = report(evidence)['items'][0]
    assert item['facts']['result_ref'] is None
    assert {m['field'] for m in item['missing_fields']} == {'description', 'weight_kg'}


@pytest.mark.parametrize('fields_order', [('all', 'price'), ('price', 'all')])
@pytest.mark.parametrize('reused_call_id', [False, True])
async def test_native_child_graph_lookup_and_submission_use_merged_source(tmp_path, monkeypatch, fields_order, reused_call_id):
    store = ContextEvidenceStore(tmp_path / 'evidence.db'); ref = await saved(store, product())
    original = deepcopy(await store.get('buyer', 'handoff', ref))
    lookup = as_langchain_tool(build_conversation_fact_lookup(store))
    choices = [{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'evidence_fields': ['description', 'weight_kg']}]
    responses = [call('conversation_fact_lookup', {'result_ref': ref, 'sku_id': 'P1003-S1', 'fields': fields})
                 for fields in fields_order]
    responses.append(submission(candidates=choices))
    _, model, dispatch, _ = factories(tmp_path, monkeypatch, responses, tools=[lookup])
    if reused_call_id:
        from tests.test_graph_recovery import ReusedIDs
        model = ReusedIDs(responses=responses)
        monkeypatch.setattr('app.application.agents.search_agent.create_chat_model', lambda *a, **kw: model)
    result = await dispatch('search_agent', task())
    assert result.ok and result.data['status'] == 'completed' and model.cursor == 3
    published = project_research_report(result.data, ExchangeRateTable(), title='真实子图报告')
    item = published['items'][0]
    assert not item['missing_fields'] and item['facts']['product']['weight_kg'] == .38
    assert item['facts']['result_ref'] == ref and item['facts']['historical'] is True
    assert await store.get('buyer', 'handoff', ref) == original


@pytest.mark.parametrize('buyer,session', [('other-buyer', 'handoff'), ('buyer', 'other-session')])
async def test_fragment_cache_cannot_create_evidence_from_an_unauthorized_lookup(tmp_path, buyer, session):
    store = ContextEvidenceStore(tmp_path / 'evidence.db'); ref = await saved(store, product())
    owner = TaskEvidence('handoff')
    await lookup_into(store, owner, ref, 'all', identifier='owner-read')
    token = ShoppingContext.set(ShoppingContextSnapshot(session, buyer, 'zh-CN', 'CNY'))
    try:
        other = TaskEvidence(session)
        data = await lookup_into(store, other, ref, 'price', identifier='other-read')
        assert data['records'] == [] and other.facts == {} and other.refs == set()
        selected = ResearchSubmission(candidates=[{'product_id': 'P1003', 'sku_id': 'P1003-S1',
                                                  'evidence_fields': ['description']}])
        assert any(e['code'] == 'candidate_unverified' for e in other.validate(selected))
        assert report(owner)['items'][0]['facts']['product']['weight_kg'] == .38
    finally:
        ShoppingContext.reset(token)
