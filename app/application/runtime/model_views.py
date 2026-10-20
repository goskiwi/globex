"""模型读资料与前端组件分离；完整记录仍由证据库和交付事件持有。"""
from copy import deepcopy

from app.infrastructure.context_products import business_view


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
    receipt = {'displayed': True, 'kind': kind}
    for key in ('result_ref', 'status', 'plan_outcome', 'unverified_requirements'):
        if key in data:
            receipt[key] = deepcopy(data[key])
    if 'hits' in data:
        receipt['sku_ids'] = [hit['default_sku_id'] for hit in data['hits']]
    if kind == 'research':
        receipt['task_ids'] = list(data['task_ids'])
    return receipt
