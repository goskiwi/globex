"""模型读资料与前端组件分离；完整记录仍由证据库和交付事件持有。"""
from copy import deepcopy

from app.infrastructure.context_products import business_view, result_identity


def product_result_view(data):
    view = business_view(data)
    if not isinstance(view, dict):
        return view
    for hit in view.get('hits', []):
        # 顶层报价属于明确的默认SKU，相同报价不再在该规格里复制一遍。
        for sku in hit.get('skus', []):
            if (hit.get('default_sku_id') == sku.get('sku_id')
                    and hit.get('landed_price') is not None
                    and hit['landed_price'] == sku.get('landed_price')):
                sku.pop('landed_price')
    for record in view.get('records', []):
        if isinstance(record.get('data'), dict):
            record['data'] = product_result_view(record['data'])
    return view


def presentation_receipt(kind, data):
    """已成功接纳的组件只回传交付事实；失败与未完成信息仍对模型可见。"""
    if kind == 'order':
        from app.application.usecases.order_outcome import order_model_receipt
        return order_model_receipt(data)
    receipt = {'displayed': True, 'kind': kind}
    for key in ('result_ref', 'status', 'plan_outcome', 'unverified_requirements'):
        if key in data:
            receipt[key] = deepcopy(data[key])
    if 'hits' in data:
        receipt['sku_ids'] = [hit['default_sku_id'] for hit in data['hits']]
    if kind == 'research':
        receipt['task_ids'] = list(data['task_ids'])
    return receipt


def historical_products(data, selected):
    """旧搜索保留候选顺序索引；完整事实只保留本次涉及的具体规格。"""
    source = product_result_view(data)
    sku_ids = {value for value in selected if '-S' in value}
    product_ids = {value for value in selected if '-S' not in value
                   and not any(sku.split('-S')[0] == value for sku in sku_ids)}
    index, retained = [], []
    for hit in source['hits']:
        index.append({'product_id': hit['product_id'], 'title': hit.get('title'),
                      'sku_ids': [sku['sku_id'] for sku in hit.get('skus', [])]})
        matching = [sku for sku in hit.get('skus', []) if sku['sku_id'] in sku_ids]
        if hit['product_id'] not in product_ids and not matching:
            continue
        if matching:
            hit['skus'] = matching
            if hit.get('default_sku_id') not in sku_ids:
                # 原顶层价可能属于另一个默认规格，不能冒充本次选中的规格。
                for key in ('default_sku_id', 'price_major', 'currency', 'source_price_major',
                            'source_currency', 'landed_price'):
                    hit.pop(key, None)
            hit['selected_sku_ids'] = [sku['sku_id'] for sku in matching]
        retained.append(hit)
    return {**{key: source[key] for key in ('result_ref', 'query_conditions', 'observed_at') if key in source},
            'historical': True, 'history_view': True, 'archived': not retained,
            'candidate_index': index, 'hits': retained}


def historical_product_views(records, selected):
    """每个SKU/报价条件只展开最近观察；旧观察仍能按各自引用恢复。"""
    seen, views = set(), {}
    for identifier, data in reversed(records):
        preview = historical_products(data, selected)
        owned = set()
        conditions = data.get('query_conditions', {})
        conditions = {**conditions, 'currency': conditions.get('target_currency') or conditions.get('currency')}
        for hit in preview['hits']:
            skus = hit.get('skus', [])
            observations = [({**hit, 'skus': [sku],
                'landed_price': sku.get('landed_price') or hit.get('landed_price', {})}, sku['sku_id'])
                for sku in skus] or [(hit, hit['product_id'])]
            for observation, target in observations:
                key = result_identity(observation, conditions)
                if key not in seen:
                    owned.add(target)
                    seen.add(key)
        views[identifier] = historical_products(data, owned)
    return views
