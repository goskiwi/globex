"""报告内容反例：不让模型重写已读事实，跨币种/规格/数量比较由代码负责。"""
from copy import deepcopy

import pytest
from pydantic import ValidationError
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.application.agents.handoff import ResearchSubmission, build_submission_tool
from app.application.tools.research_delivery_tool import _report
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
    result = project_research_report(data, RATES)
    assert result['product_count'] == 2 and result['sku_count'] == 2
    unit, landed = result['price_comparisons']
    assert unit['minimum_minor'] == 29118 and unit['maximum_minor'] == 29900
    assert unit['spread_minor'] == 782 and unit['lowest'][0]['sku_id'] == 'P40018-S1'
    assert landed['minimum_minor'] == 31618 and landed['maximum_minor'] == 32400
    text = _report(result)
    assert '2款商品、2个明确规格' in text and '3款' not in text and '头戴式' not in text
    assert result['items'][0]['open_questions'] == ['是否有电脑仓？']
    assert '商品原文' not in text and '取舍建议' not in text
    assert data == before, '事实投影不能改写原始回执'


def test_assessment_is_presented_separately_without_replacing_facts_or_price_comparisons():
    data = report_pair()
    candidate = data['candidates'][0]
    candidate['assessment'] = '适合优先减轻随身负担的人；背负舒适度尚未核验。'
    before = deepcopy(candidate['facts'])
    result = project_research_report(data, RATES)
    assert result['items'][0]['assessment'] == candidate['assessment']
    assert result['items'][0]['facts'] == before
    text = _report(result)
    assert result['items'][0]['facts']['product']['description'] == '支持蓝牙和降噪，未提供佩戴形式'
    assert result['items'][0]['facts']['sku']['price_major'] == 299
    assert result['price_comparisons'][0]['minimum_minor'] == 29118
    assert candidate['assessment'] not in text, '取舍由组件完整展示，正文不重复'
    assert '2款商品、2个明确规格' in text


def test_submission_requires_current_assessment_without_filling_old_payload():
    schema = convert_to_openai_tool(build_submission_tool(ResearchSubmission))['function']['parameters']
    assert 'assessment' in schema['properties']['candidates']['items']['required']
    with pytest.raises(ValidationError) as failure:
        ResearchSubmission(status='completed', candidates=[{'product_id': 'P1022'}])
    assert any(e['loc'] == ('candidates', 0, 'assessment') and e['type'] == 'missing'
               for e in failure.value.errors())


def test_counts_distinguish_canonical_products_and_explicit_skus():
    data = report_pair()
    first, second = data['candidates']
    first['facts']['product']['canonical_product_id'] = 'same-product'
    second['facts']['product']['canonical_product_id'] = 'same-product'
    result = project_research_report(data, RATES)
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
    comparisons = project_research_report(data, RATES)['price_comparisons']
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
    result = project_research_report(data, RATES)
    assert result['price_comparisons'] == []
    assert result['items'][1]['facts']['historical'] is historical


def test_tied_prices_keep_all_lowest_candidates_and_full_read_facts():
    data = report_pair()
    data['candidates'][1]['facts']['sku'].update(price_major=299, currency='CNY')
    product = data['candidates'][0]['facts']['product']
    product['catalog_extension'] = {'value': '保留原始目录字段'}
    before = deepcopy(product)
    result = project_research_report(data, RATES)
    unit = result['price_comparisons'][0]
    assert unit['spread_minor'] == 0 and len(unit['lowest']) == len(unit['highest']) == 2
    assert result['items'][0]['facts']['product'] == before
    assert result['items'][0]['facts']['product']['weight_kg'] == .38
    assert 'evidence' not in result['items'][0]


@pytest.mark.parametrize('field,value', [('summary', '最便宜'), ('unknowns', ['没电脑仓']),
                                         ('reason', '头戴式'), ('evidence_fields', ['weight_kg'])])
def test_removed_prose_inputs_are_absent_from_native_schema(field, value):
    schema = convert_to_openai_tool(build_submission_tool(ResearchSubmission))['function']['parameters']
    assert 'summary' not in schema['properties'] and 'unknowns' not in schema['properties']
    properties = schema['properties']['candidates']['items']['properties']
    assert 'reason' not in properties and 'evidence_fields' not in properties
    payload = {'status': 'completed', 'candidates': [
        {'product_id': 'P1022', 'assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}]}
    if field in {'reason', 'evidence_fields'}:
        payload['candidates'][0][field] = value
    else:
        payload[field] = value
    with pytest.raises(ValidationError):
        ResearchSubmission.model_validate(payload)


def test_model_cannot_inject_facts_or_duplicate_candidate():
    candidate = {'product_id': 'P1022', 'assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}
    with pytest.raises(ValidationError):
        ResearchSubmission(status='completed', candidates=[{**candidate, 'facts': {'product': {'weight_kg': .01}}}])
    assert ResearchSubmission(status='completed', candidates=[candidate]).candidates[0].product_id == 'P1022'
    with pytest.raises(ValidationError, match='不能重复'):
        ResearchSubmission(status='completed', candidates=[candidate, candidate])
