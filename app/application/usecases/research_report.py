"""研究报告从已读证据生成参数、比较和字段缺口；模型只负责选择。"""
import math
from datetime import datetime
from app.application.research_evidence import candidate_evidence
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


def _observation_time(facts):
    """只有带时区的实际观察时间才能用于跨来源的新旧判断。"""
    value = facts.get('observed_at')
    if not isinstance(value, str):
        return None
    try:
        observed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return observed if observed.tzinfo is not None else None


def _current_product_observations(candidates):
    """先限定当前来源，再选最新观察；无时间或同时间的来源保留作一致性比较。"""
    products = {}
    for candidate in candidates:
        products.setdefault(candidate['product_id'], []).append(candidate['facts'])
    for identifier, facts in products.items():
        current = [fact for fact in facts if fact.get('historical') is False]
        times = [_observation_time(fact) for fact in current]
        if current and all(time is not None for time in times):
            latest = max(times)
            current = [fact for fact, time in zip(current, times) if time == latest]
        products[identifier] = current
    return products


def report_metrics(candidates):
    products = _current_product_observations(candidates)
    comparisons=[]
    for field,label,unit,path in [
        ('weight_kg','重量','kg',('weight_kg',)),
        ('rating_average','目录评分','分',('rating_summary','average')),
        ('review_count','目录评价数','条',('rating_summary','review_count')),
    ]:
        rows,excluded=[],[]
        # 排序仅稳定展示顺序，不参与来源选择或冲突消解。
        for identifier in sorted(products):
            observations=products[identifier]
            if not observations:
                excluded.append({'product_id':identifier,'reason':'historical_only'})
                continue
            values=[]
            for facts in observations:
                value=facts['product']
                for key in path:
                    value=value.get(key) if isinstance(value,dict) else None
                if type(value) in (int,float) and math.isfinite(value):
                    values.append(value)
            # 同时点或时间不明的不同来源必须一致；缺失不能用另一快照的值补齐。
            reason = ('conflicting_observations' if len(set(values))>1 else
                      'missing_value' if len(values)!=len(observations) else None)
            if reason:
                excluded.append({'product_id':identifier,'reason':reason})
                continue
            titles={fact['product'].get('title') for fact in observations if fact['product'].get('title')}
            title=next(iter(titles)) if len(titles)==1 else identifier
            rows.append({'product_id':identifier,'title':title,'value':values[0]})
        comparable = len(rows) >= 2
        low = min(r['value'] for r in rows) if comparable else None
        high = max(r['value'] for r in rows) if comparable else None
        comparisons.append({'field':field,'label':label,'unit':unit,'compared_product_count':len(rows),
            'excluded_products':excluded,'minimum':low,'maximum':high,
            'lowest':[r for r in rows if r['value']==low] if comparable else [],
            'highest':[r for r in rows if r['value']==high] if comparable else []})
    return comparisons


def project_research_report(data, rates, *, title):
    candidates=data['candidates']
    caveats=['比较范围仅为本报告列出的候选；目录参数未经实物验证，缺失不表示没有该能力。']
    if any(c['facts'].get('historical') for c in candidates):
        caveats.append('包含历史资料，历史候选不纳入当前价格和参数排名。')
    if any(c['facts']['product'].get('rating_is_live') is not True for c in candidates):
        caveats.append('目录评分与评价数量不是实时评价，不代表实际使用效果。')
    return {'task_id': data['scope']['task_id'], 'title': title, 'status': data['status'],
            **research_counts(candidates), 'price_comparisons': report_prices(data, rates),
            'metric_comparisons':report_metrics(candidates),
            'items':[candidate_evidence(c) for c in candidates], 'caveats':caveats,
            **{key: data[key] for key in ('unmet_requirements', 'questions', 'issues')},
            'transaction_state': data['transaction_state']}
