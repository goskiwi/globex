"""原生工具消息的请求局部无损共享；不改 checkpoint 或证据。"""
from copy import deepcopy
import json
from langchain_core.messages import ToolMessage
from app.infrastructure.context_products import business_view, result_identity, token_estimate
from app.application.runtime.results import message_data, project_data


def project_skill_history(messages):
    """请求局部投影：旧正文不进入新请求；当前正文由 Skill 装配器按引用注入。"""
    projected = []
    for message in messages:
        if message.name in {"skill_catalog", "skill_reference", "selected_skill_reference"}:
            continue
        if isinstance(message, ToolMessage) and message.name == "load_agent_skill_tool" and message.status == "success":
            message = message.model_copy(update={
                "content": "Skill 加载回执已保留；仅本轮 skill_reference 是当前有效正文。",
                "artifact": None})
        projected.append(message)
    return projected


def read_output(message):
    if not isinstance(message, ToolMessage):return None
    payload = message_data(message)
    return payload if isinstance(payload,dict) else None


def share_identical_products(messages, *, compact_rules=False):
    prepared=deepcopy(messages);seen={}
    for message in prepared:
        original=read_output(message)
        if not original:continue
        # 所有结构化工具回执共用模型投影，推荐/比较和回查不能重新带入展示或目录管理字段。
        payload=business_view(original)
        projected=payload!=original
        shared=False
        if isinstance(payload.get('hits'),list) and not payload.get('archived'):
            for position,hit in enumerate(payload['hits'],1):
                if not isinstance(hit,dict) or 'same_business_fields_as' in hit:continue
                # score 属于本次查询，不是商品事实；它变化不能使相同资料重复进入请求。
                business_fields={k:v for k,v in hit.items() if k != 'score'}
                key=(result_identity(hit,payload.get('query_conditions',{})),json.dumps(business_fields,ensure_ascii=False,sort_keys=True))
                target=seen.get(key)
                if target and token_estimate(hit)>180:
                    payload['hits'][position-1]={'product_id':hit.get('product_id'),'same_business_fields_as':target}
                    if 'score' in hit:
                        payload['hits'][position-1]['score']=hit['score']
                    shared=True
                else:seen[key]={'tool_call_id':message.tool_call_id,'position':position}
        if shared:
            payload['shared_fields_notice']='引用对象的商品业务字段就在本次输入对应工具结果中；score 不继承引用对象，本批分数、顺序、查询条件、观察时间仍以本条为准。'
        if compact_rules:
            payload.pop('shared_fields_notice',None)
            if payload.get('archived'):payload.pop('notice',None)
        if shared or projected or compact_rules:
            rendered=project_data(message,payload)
            message.content, message.artifact = rendered.content, rendered.artifact
    return prepared


EVIDENCE_RULES = ('archived=true 表示历史结果已归档，按 result_ref 回查，不能当作当前报价。'
    'same_business_fields_as 只引用本次输入内完整的相同商品业务字段，不继承引用对象的score；本批分数、顺序、查询条件、观察时间保持各自来源。'
    '最新用户消息后的权威工具结果才是当前核验，不同 SKU、目的地、币种与数量不得混用；缺失字段说明未知。')
