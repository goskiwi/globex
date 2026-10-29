"""研究者只选择证据字段；展示值和缺口从已读商品资料生成。"""
from copy import deepcopy
from typing import Literal

ResearchEvidenceField = Literal['description', 'highlights', 'material_tags', 'weight_kg',
    'dimensions_cm', 'package_dimensions_cm', 'rating_summary']

EVIDENCE_LABELS = {'description':'目录描述', 'highlights':'目录亮点', 'material_tags':'材质标签',
    'weight_kg':'重量（kg）', 'dimensions_cm':'商品尺寸（cm）',
    'package_dimensions_cm':'包装尺寸（cm）', 'rating_summary':'目录评分与评价数'}


def candidate_evidence(candidate):
    product = candidate['facts']['product']
    fields = list(dict.fromkeys(candidate['evidence_fields']))
    evidence, missing = [], []
    for field in fields:
        value = product.get(field)
        row = {'field':field, 'label':EVIDENCE_LABELS[field]}
        if value is None or value == '' or isinstance(value,(dict,list)) and not value:
            missing.append(row)
        else:
            evidence.append({**row,'value':deepcopy(value)})
    return {'product_id':candidate['product_id'], 'sku_id':candidate.get('sku_id'),
            'evidence_fields':fields, 'evidence':evidence, 'missing_fields':missing,
            'facts':deepcopy(candidate['facts'])}
