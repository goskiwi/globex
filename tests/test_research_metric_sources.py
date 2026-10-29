"""真实目录读取、证据回查和研究交付：参数聚合不得依赖候选排列。"""
from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from app.application.agents.handoff import ResearchSubmission
from app.application.runtime.handoff import TaskEvidence
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from app.application.tools.research_delivery_tool import build_research_delivery_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.sku import Sku
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


class ResearchInputs:
    def __init__(self, folder):
        self.store = ContextEvidenceStore(folder / 'observations.db')
        self.lookup = build_conversation_fact_lookup(self.store)
        self.evidence = TaskEvidence('metrics-session')
        self.bus = TradeEventBus()
        self.skus = []

    async def read(self, sku_id, *, weight=.4, reviews=287, observed_at='2026-10-22T10:00:00+00:00',
                   historical=False, title=None):
        product_id = sku_id.split('-S')[0]
        product = Product(product_id=product_id, title=title or product_id+' 合成商品', brand='合成',
            category='旅行装备', origin_country='CN', description='仅用于来源组合验证的合成资料',
            skus=[Sku(sku_id, '合成规格', Money.from_major_units(100, 'CNY'), 5)],
            weight_kg=weight, rating_summary={'average':4.5, 'review_count':reviews})
        catalog = CatalogSearchUseCase(InMemoryProductRepository([product]))
        result = await catalog.execute(ProductSearchSpec(sku_id=sku_id))
        result['observed_at'] = observed_at
        ref = await self.store.save('metrics-buyer', 'metrics-session', 'products', result)
        if historical:
            returned = await self.lookup(result_ref=ref, sku_id=sku_id, fields='all')
            assert returned.ok
            self.evidence.capture_lookup([ToolMessage(content='', name='conversation_fact_lookup',
                tool_call_id=ref, artifact={'data':returned.data})])
        else:
            with observe_run_events(self.evidence.capture):
                self.bus.publish('metrics-session', 'tool.result',
                    {'tool':'product_search_tool', **result, 'result_ref':ref})
        self.skus.append(sku_id)
        return ref

    async def present(self, order):
        submission = ResearchSubmission(candidates=[{'product_id':sku.split('-S')[0], 'sku_id':sku,
            'evidence_fields':['weight_kg','rating_summary']} for sku in order])
        assert not self.evidence.validate(submission)
        data = self.evidence.delivered(submission).model_dump()
        data.update(scope={'task_id':'metrics', 'effective_search':{'parameters':{
            'target_currency':'CNY', 'ship_to':'CN'}}}, transaction_state='none')
        runtime = SimpleNamespace(state={'messages':[ToolMessage(content='', name='task_dispatch',
            tool_call_id='dispatch', artifact={'data':data})],
            'task_plan':{'tasks':{'metrics':{'stale':False, 'scope':data['scope']}}}})
        result = await build_research_delivery_tool(ExchangeRateTable())(
            [{'task_id':'metrics', 'title':'合成来源研究'}], runtime)
        assert result.ok
        return result.data['reports'][0]


@pytest.fixture
def research_inputs(tmp_path):
    token = ShoppingContext.set(ShoppingContextSnapshot('metrics-session','metrics-buyer','zh-CN','CNY'))
    try:
        yield ResearchInputs(tmp_path)
    finally:
        ShoppingContext.reset(token)


async def metric_sets(inputs):
    before = deepcopy(inputs.evidence.facts)
    reports = [await inputs.present(order) for order in permutations(inputs.skus)]
    first = reports[0]['metric_comparisons']
    assert first and all(report['metric_comparisons']==first for report in reports)
    assert inputs.evidence.facts == before
    # 原始候选展示仍遵从模型选择顺序，只有比较集合消除排列影响。
    assert [item['sku_id'] for item in reports[0]['items']] == inputs.skus
    return {row['field']:row for row in first}


async def test_current_and_historical_skus_of_one_product_have_order_independent_metrics(research_inputs):
    inputs = research_inputs
    await inputs.read('P1003-S2',weight=.9,reviews=999,observed_at='2026-09-01T00:00:00+00:00',historical=True)
    await inputs.read('P1003-S1',weight=.4,reviews=287)
    await inputs.read('P1008-S1',weight=.6,reviews=349)
    metrics = await metric_sets(inputs)
    assert set(metrics)=={'weight_kg','rating_average','review_count'}
    assert metrics['weight_kg']['minimum']==.4 and metrics['weight_kg']['maximum']==.6
    assert metrics['review_count']['maximum']==349
    assert all(row['compared_product_count']==2 and row['excluded_products']==[] for row in metrics.values())


@pytest.mark.parametrize('reverse_capture',[False,True])
async def test_latest_observation_wins_independently_of_capture_and_display_order(research_inputs,reverse_capture):
    inputs = research_inputs
    snapshots=[('P1003-S1',.4,'2026-10-22T12:00:00+02:00'),
               ('P1003-S2',.8,'2026-10-22T10:01:00+00:00')]
    for sku,weight,observed in reversed(snapshots) if reverse_capture else snapshots:
        await inputs.read(sku,weight=weight,observed_at=observed)
    await inputs.read('P1008-S1',weight=.6)
    metrics = await metric_sets(inputs)
    assert metrics['weight_kg']['maximum']==.8
    assert metrics['weight_kg']['highest'][0]['product_id']=='P1003'


async def test_newer_missing_value_does_not_borrow_older_value(research_inputs):
    inputs = research_inputs
    await inputs.read('P1003-S1',weight=.4,observed_at='2026-10-22T10:00:00+00:00')
    await inputs.read('P1003-S2',weight=None,observed_at='2026-10-22T10:01:00+00:00')
    await inputs.read('P1008-S1',weight=.6)
    await inputs.read('P1009-S1',weight=.6)
    metric = (await metric_sets(inputs))['weight_kg']
    assert metric['excluded_products']==[{'product_id':'P1003','reason':'missing_value'}]
    assert {row['product_id'] for row in metric['lowest']}=={'P1008','P1009'}


@pytest.mark.parametrize('second_time',[
    '2026-10-22T12:00:00+02:00', None, '时间未知', '2026-10-22',
])
async def test_concurrent_or_unordered_conflicts_are_excluded_per_metric(research_inputs,second_time):
    inputs = research_inputs
    await inputs.read('P1003-S1',weight=.4,reviews=287)
    await inputs.read('P1003-S2',weight=.8,reviews=287,observed_at=second_time)
    await inputs.read('P1008-S1',weight=.6,reviews=349)
    await inputs.read('P1009-S1',weight=.6,reviews=349)
    metrics = await metric_sets(inputs)
    assert metrics['weight_kg']['excluded_products']==[
        {'product_id':'P1003','reason':'conflicting_observations'}]
    assert metrics['review_count']['compared_product_count']==3
    assert metrics['review_count']['excluded_products']==[]
    assert [r['product_id'] for r in metrics['review_count']['highest']]==['P1008','P1009']


async def test_unordered_equal_observations_can_be_compared_without_arbitrary_title(research_inputs):
    inputs = research_inputs
    await inputs.read('P1003-S1',weight=.4,title='同一商品观察甲')
    await inputs.read('P1003-S2',weight=.4,title='同一商品观察乙',observed_at=None)
    await inputs.read('P1008-S1',weight=.6)
    metrics = await metric_sets(inputs)
    assert metrics['weight_kg']['excluded_products']==[]
    assert metrics['weight_kg']['lowest']==[{'product_id':'P1003','title':'P1003','value':.4}]


async def test_history_and_missing_values_keep_distinct_reasons(research_inputs):
    inputs = research_inputs
    await inputs.read('P1003-S1',weight=.9,historical=True)
    await inputs.read('P1008-S1',weight=None)
    await inputs.read('P1009-S1',weight=.6)
    await inputs.read('P1010-S1',weight=.6)
    metric = (await metric_sets(inputs))['weight_kg']
    assert metric['excluded_products']==[
        {'product_id':'P1003','reason':'historical_only'},
        {'product_id':'P1008','reason':'missing_value'}]
    assert metric['minimum']==metric['maximum']==.6


async def test_two_products_with_one_conflict_explain_why_ranking_is_unavailable(research_inputs):
    inputs = research_inputs
    await inputs.read('P1003-S1',weight=.4)
    await inputs.read('P1003-S2',weight=.8)
    await inputs.read('P1008-S1',weight=.6)
    metrics = await metric_sets(inputs)
    weight = metrics['weight_kg']
    assert weight['compared_product_count']==1
    assert weight['minimum'] is weight['maximum'] is None
    assert weight['lowest']==weight['highest']==[]
    assert weight['excluded_products']==[{'product_id':'P1003','reason':'conflicting_observations'}]
    assert metrics['rating_average']['compared_product_count']==2


async def test_two_products_with_one_historical_observation_never_publish_single_item_extrema(research_inputs):
    inputs = research_inputs
    await inputs.read('P1003-S1',weight=.9,historical=True)
    await inputs.read('P1008-S1',weight=.6)
    metrics = await metric_sets(inputs)
    assert all(metric['compared_product_count']==1 and metric['minimum'] is metric['maximum'] is None
               and metric['lowest']==metric['highest']==[] for metric in metrics.values())
    assert all(metric['excluded_products']==[{'product_id':'P1003','reason':'historical_only'}]
               for metric in metrics.values())
