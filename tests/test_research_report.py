"""报告内容反例：不让模型重写已读事实，跨币种/规格/数量比较由代码负责。"""
from copy import deepcopy

import pytest
from pydantic import ValidationError
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.application.agents.handoff import ResearchSubmission, build_submission_tool
from app.application.usecases.research_report import project_research_report, research_summary
from app.domain.catalog.exchange_rate import ExchangeRateTable
from tests.test_research_delivery_boundaries import research_receipt


RATES = ExchangeRateTable()


def report_pair():
    data = research_receipt('P1022', description='支持蓝牙和降噪，未提供佩戴形式')
    other = research_receipt('P40018')['candidates'][0]
    data['candidates'].append(other)
    first = data['candidates'][0]['facts']
    first['sku'].update(price_major=299, currency='CNY')
    first['quotes'][0]['line']['total_amount_minor'] = 32400
    other['facts']['sku'].update(price_major=54.94, currency='SGD')
    other['facts']['quotes'][0]['line']['total_amount_minor'] = 31618
    return data


def test_report_ignores_model_summary_and_computes_cross_currency_ranking():
    data = report_pair()
    data['summary'] = '3款头戴式耳机，P1022是最低价'
    before = deepcopy(data)
    result = project_research_report(data, RATES, title='研究结果')
    assert result['product_count'] == 2 and result['sku_count'] == 2
    unit, landed = result['price_comparisons']
    assert unit['minimum_minor'] == 29118 and unit['maximum_minor'] == 29900
    assert unit['spread_minor'] == 782 and unit['lowest'][0]['sku_id'] == 'P40018-S1'
    assert landed['minimum_minor'] == 31618 and landed['maximum_minor'] == 32400
    assert 'summary' not in result
    assert result['items'][0]['evidence'][0]['value'] == '支持蓝牙和降噪，未提供佩戴形式'
    assert data == before, '事实投影不能改写原始回执'


def test_evidence_fields_resolve_values_and_missing_fields_without_model_prose():
    data=report_pair()
    candidate=data['candidates'][0]
    candidate['evidence_fields']=['weight_kg','dimensions_cm','package_dimensions_cm','description']
    before=deepcopy(candidate['facts'])
    result=project_research_report(data,RATES,title='研究结果')
    item=result['items'][0]
    assert item['facts']==before and item['evidence'][0]['value']==.38
    assert item['evidence'][1]['label']=='商品尺寸（cm）'
    assert item['missing_fields']==[{'field':'package_dimensions_cm','label':'包装尺寸（cm）'}]
    assert 'assessment' not in item and 'caveats' not in item
    assert result['price_comparisons'][0]['minimum_minor']==29118


def test_submission_requires_field_selection_without_old_prose_compatibility():
    schema=convert_to_openai_tool(build_submission_tool(ResearchSubmission))['function']['parameters']
    assert 'evidence_fields' in schema['properties']['candidates']['items']['required']
    with pytest.raises(ValidationError):ResearchSubmission(candidates=[{'product_id':'P1022'}])
    with pytest.raises(ValidationError):ResearchSubmission(candidates=[{'product_id':'P1022','evidence_fields':['invented']}])


def test_counts_distinguish_canonical_products_and_explicit_skus():
    data = report_pair()
    first, second = data['candidates']
    first['facts']['product']['canonical_product_id'] = 'same-product'
    second['facts']['product']['canonical_product_id'] = 'same-product'
    result = project_research_report(data, RATES, title='研究结果')
    assert (result['product_count'], result['sku_count']) == (1, 2)
    assert '1款商品、2个明确规格' in research_summary('completed', data['candidates'])


@pytest.mark.parametrize('change', ['missing', 'sku', 'quantity', 'destination', 'currency', 'historical'])
def test_incomparable_quotes_never_create_landed_ranking(change):
    data = report_pair()
    facts = data['candidates'][1]['facts']
    quote = facts['quotes'][0]
    if change == 'missing':
        facts['quotes'] = []
    elif change == 'sku':
        quote['line']['sku_id'] = 'P40018-S2'
    elif change == 'quantity':
        quote['line']['quantity'] = 2
    elif change == 'destination':
        quote['ship_to'] = 'US'
    elif change == 'currency':
        quote['currency'] = quote['line']['currency'] = 'USD'
    else:
        quote['historical'] = True
    comparisons = project_research_report(data, RATES, title='研究结果')['price_comparisons']
    assert len(comparisons) == 1 and comparisons[0]['kind'] == 'unit_price'


@pytest.mark.parametrize('historical', [False, True])
def test_product_only_or_historical_candidates_do_not_borrow_current_sku_prices(historical):
    data = report_pair()
    candidate = data['candidates'][1]
    if historical:
        candidate['facts']['historical'] = True
    else:
        candidate['sku_id'] = None
        candidate['facts']['sku'] = None
    result = project_research_report(data, RATES, title='研究结果')
    assert result['price_comparisons'] == []
    assert result['items'][1]['facts']['historical'] is historical


def test_tied_prices_keep_all_lowest_candidates_and_full_read_facts():
    data = report_pair()
    data['candidates'][1]['facts']['sku'].update(price_major=299, currency='CNY')
    product = data['candidates'][0]['facts']['product']
    product['catalog_extension'] = {'value': '保留原始目录字段'}
    before = deepcopy(product)
    result = project_research_report(data, RATES, title='研究结果')
    unit = result['price_comparisons'][0]
    assert unit['spread_minor'] == 0 and len(unit['lowest']) == len(unit['highest']) == 2
    assert result['items'][0]['facts']['product'] == before
    assert result['items'][0]['facts']['product']['weight_kg'] == .38
    assert result['items'][0]['evidence'][1]['value']==.38


@pytest.mark.parametrize('field,value',[('summary','最便宜'),('unknowns',['没电脑仓']),
    ('assessment','54g，评价数最多'),('caveats',['没有电脑仓']),('reason','头戴式')])
def test_removed_prose_inputs_are_absent_from_native_schema(field,value):
    schema=convert_to_openai_tool(build_submission_tool(ResearchSubmission))['function']['parameters']
    candidate={'product_id':'P1022','evidence_fields':['weight_kg']}
    payload={'candidates':[candidate]}
    if field in {'assessment','reason'}:candidate[field]=value
    else:payload[field]=value
    assert 'assessment' not in schema['properties']['candidates']['items']['properties']
    assert 'caveats' not in schema['properties']
    with pytest.raises(ValidationError):ResearchSubmission.model_validate(payload)


def test_model_cannot_inject_facts_or_duplicate_candidate():
    candidate = {'product_id': 'P1022', 'evidence_fields':['description','weight_kg']}
    with pytest.raises(ValidationError):
        ResearchSubmission(candidates=[{**candidate, 'facts': {'product': {'weight_kg': .01}}}])
    assert ResearchSubmission(candidates=[candidate]).candidates[0].product_id == 'P1022'
    with pytest.raises(ValidationError, match='不能重复'):
        ResearchSubmission(candidates=[candidate, candidate])


def test_metric_rankings_compute_349_over_287_and_never_mix_missing_or_historical():
    data=report_pair()
    first,second=data['candidates']
    first['facts']['product']['rating_summary']={'average':4.8,'review_count':287}
    first['facts']['product']['weight_kg']=.055
    second['facts']['product']['rating_summary']={'average':4.2,'review_count':349}
    second['facts']['product']['weight_kg']=.04
    metrics={m['field']:m for m in project_research_report(data,RATES,title='证据')['metric_comparisons']}
    assert metrics['review_count']['maximum']==349
    assert metrics['review_count']['highest'][0]['product_id']==second['product_id']
    assert metrics['weight_kg']['maximum']==.055
    assert metrics['rating_average']['highest'][0]['product_id']==first['product_id']
    missing=deepcopy(second);missing['product_id']='missing';missing['facts']['product']={}
    data['candidates'].append(missing)
    metric=next(m for m in project_research_report(data,RATES,title='证据')['metric_comparisons'] if m['field']=='review_count')
    assert metric['compared_product_count']==2 and metric['excluded_products']==[{'product_id':'missing','reason':'missing_value'}]
    second['facts']['historical']=True
    unavailable=project_research_report(data,RATES,title='证据')['metric_comparisons']
    assert all(m['compared_product_count']==1 and m['minimum'] is m['maximum'] is None
               and m['lowest']==m['highest']==[] for m in unavailable)
    assert all(m['excluded_products']==[{'product_id':'P40018','reason':'historical_only'},
                                        {'product_id':'missing','reason':'missing_value'}] for m in unavailable)
