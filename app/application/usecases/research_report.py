"""研究报告事实投影：只读证据选字段，数量及价格比较由业务代码计算。"""
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money


def research_counts(candidates):
    products = {(c.get('facts', {}).get('product', {}).get('canonical_product_id') or c['product_id'])
                for c in candidates}
    skus = {c['sku_id'] for c in candidates if c.get('sku_id')}
    return {'product_count': len(products), 'sku_count': len(skus)}


def research_summary(status, candidates):
    counts = research_counts(candidates)
    caption = {'completed':'研究完成', 'partial':'研究部分完成', 'needs_input':'研究需要补充信息',
               'failed':'研究未完成'}[status]
    return f"{caption}：交回{counts['product_count']}款商品、{counts['sku_count']}个明确规格。"


FIELD_LABELS = {'description':'商品原文', 'highlights':'目录亮点', 'dimensions_cm':'商品尺寸（厘米）',
                'material_tags':'目录材质', 'rating_summary':'目录评分', 'ships_to':'目录配送范围'}


def selected_evidence(candidate):
    """模型只能选已存在的字段，不能把新属性或解释写成商品事实。"""
    product = candidate['facts']['product']
    return [{'field': key, 'label': FIELD_LABELS[key], 'value': product[key]}
            for key in dict.fromkeys(candidate['evidence_fields']) if product.get(key) not in (None, '', [], {})]


def _price_comparison(rows, **conditions):
    amounts = [row['amount_minor'] for row in rows]
    low, high = min(amounts), max(amounts)
    return {**conditions, 'compared_sku_count': len(rows),
            'minimum_minor': low, 'maximum_minor': high, 'spread_minor': high-low,
            'lowest': [row for row in rows if row['amount_minor']==low],
            'highest': [row for row in rows if row['amount_minor']==high]}


def report_prices(data, rates: ExchangeRateTable):
    """单价复用业务汇率；到手价只比较同目的地/数量/币种的完整已读报价。"""
    candidates = data['candidates']
    if len(candidates) < 2:
        return []
    policy = data['scope']['effective_search']['parameters']
    currency, ship_to = policy['target_currency'], policy.get('ship_to')
    unit_rows, quotes_by_candidate = [], []
    for candidate in candidates:
        facts, sku_id = candidate['facts'], candidate.get('sku_id')
        sku = facts.get('sku')
        # 历史资料不能使用当前汇率冒充同一历史时点的比较；未选规格也不借默认SKU金额。
        if not sku_id or not sku or sku['sku_id']!=sku_id or facts.get('historical') is not False:
            return []
        price = rates.convert(Money.from_major_units(sku['price_major'], sku['currency']), currency)
        identity = {'product_id': candidate['product_id'], 'sku_id': sku_id,
                    'title': facts['product'].get('title') or candidate['product_id']}
        unit_rows.append({**identity, 'amount_minor': price.amount_in_minor_units})
        quotes = {}
        for quote in facts.get('quotes', []):
            line = quote['line']
            if (quote.get('historical') is False and quote['currency']==currency and quote['ship_to']==ship_to
                    and line['product_id']==candidate['product_id'] and line['sku_id']==sku_id
                    and line['currency']==currency):
                quotes[(line['quantity'], quote['ship_to'], currency)] = {
                    **identity, 'amount_minor': line['total_amount_minor']}
        quotes_by_candidate.append(quotes)
    if not unit_rows:
        return []
    comparisons = [_price_comparison(unit_rows, kind='unit_price', quantity=1, currency=currency)]
    common = set(quotes_by_candidate[0]).intersection(*(set(q) for q in quotes_by_candidate[1:]))
    for quantity, destination, target in sorted(common):
        key = (quantity, destination, target)
        comparisons.append(_price_comparison([q[key] for q in quotes_by_candidate],
            kind='landed_price', quantity=quantity, ship_to=destination, currency=target))
    return comparisons


def project_research_report(data, rates):
    return {'task_id': data['scope']['task_id'], 'goal': data['scope']['goal'], 'status': data['status'],
            **research_counts(data['candidates']), 'price_comparisons': report_prices(data, rates),
            'items': [{**candidate, 'evidence': selected_evidence(candidate)} for candidate in data['candidates']],
            **{key: data[key] for key in ('unmet_constraints', 'questions', 'issues')},
            'transaction_state': data['transaction_state']}
