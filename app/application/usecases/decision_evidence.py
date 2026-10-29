"""推荐选择由模型提交；展示依据、比较结果和费用说明由实际商品与报价生成。"""
from copy import deepcopy
from typing import Literal

from app.application.research_evidence import EVIDENCE_LABELS

DecisionDimension = Literal['price','weight_kg','rating_summary','material_tags','dimensions_cm',
                            'description','highlights','spec','requirements']
DIMENSION_LABELS = {'price':'本次报价','spec':'规格','requirements':'需求核验',**EVIDENCE_LABELS}


def readable(field,value):
    if isinstance(value,list):return '、'.join(map(str,value))
    if field in {'dimensions_cm','package_dimensions_cm'}:
        return '、'.join(label+str(value[key])+'厘米' for key,label in
                        (('length','长'),('width','宽'),('height','高')) if key in value)
    if field=='rating_summary':
        return str(value.get('average','未知'))+'分，'+str(value.get('review_count','未知'))+'条评价'
    return str(value)


def attach_decision_evidence(card, fields, requirements):
    """缺失和用途适配各自表达；有外形尺寸不等于已核实内部适配。"""
    evidence=[];missing=[]
    for field in dict.fromkeys(fields):
        value=card.get(field)
        if value is None or value=='' or isinstance(value,(dict,list)) and not value:
            missing.append(EVIDENCE_LABELS[field])
        else:evidence.append({'field':field,'label':EVIDENCE_LABELS[field],'value':deepcopy(value)})
    card['decision_evidence']=evidence
    card['requirement_checks']=[{'requirement':text,'status':'unverified'} for text in dict.fromkeys(requirements)]
    card['recommendation_reason']='；'.join(row['label']+'：'+readable(row['field'],row['value']) for row in evidence) or '已核对当前规格、库存与价格。'
    card['tradeoffs']=(['目录未提供：'+'、'.join(missing)] if missing else [])
    if requirements:card['tradeoffs'].append('以下要求尚无核验结论：'+'；'.join(dict.fromkeys(requirements)))


def decision_guidance(cards, *, dimensions=(), preferred=None, bundle=False):
    """同一业务结果同时用于最终回复和组件，不再另写一份金额或能力结论。"""
    first=next((c for c in cards if c['default_sku_id']==preferred),cards[0])
    pending=list(dict.fromkeys(check['requirement'] for c in cards for check in c['requirement_checks']))
    text=(f'已整理{len(cards)}个规格的组合。' if bundle else
          f'已整理{len(cards)}个候选规格，优先'+('核实' if pending else '考虑')+first['title']+'。')
    if dimensions:text+='比较项：'+'、'.join(DIMENSION_LABELS[d] for d in dict.fromkeys(dimensions))+'。'
    if all('landed_price' in c for c in cards):
        text+='金额按'+('组合合计' if bundle else '各选项独立')+'核算，包含本次报价中的运费与税费。'
    else:text+='当前只展示商品价，到手金额尚未核算。'
    if pending:text+='以下要求尚未核实：'+'；'.join(pending)+'。请把这些商品作为待核实候选。'
    return text
