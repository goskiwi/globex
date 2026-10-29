"""记忆操作的业务结果与完成说明共用同一数据，不由模型重新解释写入状态。"""
from dataclasses import asdict
from app.application.runtime.results import ToolResult


def memory_outcome(action, status, *, facts=(), created_ids=(), statement='', remaining=()):
    records=[{k:v for k,v in asdict(p).items() if k!='buyer_id'} for p in facts]
    labels='；'.join(p.statement for p in facts) or statement
    if action=='save':
        new=[p.statement for p in facts if p.memory_id in created_ids]
        existing=[p.statement for p in facts if p.memory_id not in created_ids]
        lines=[]
        if new:lines.append('已保存长期偏好：'+'；'.join(new)+'。')
        if existing:lines.append('以下偏好已存在，本次没有重复新增：'+'；'.join(existing)+'。')
        guidance='\n'.join(lines)
    elif status=='not_found':
        guidance='未找到指定偏好，没有修改或删除任何内容。'
        guidance+=('当前偏好：'+'；'.join(p.statement for p in remaining)+'。') if remaining else '当前没有任何长期偏好。'
    else:
        guidance=('已更新长期偏好：' if action=='update' else '已撤回长期偏好：')+labels+'。'
    return ToolResult({'operation':'memory','action':action,'status':status,
        'preferences':records,'created_ids':list(created_ids),'guidance':guidance})


def memory_completion(receipts):
    """只描述已返回的执行结果；拒绝及未知失败保留自身语义。"""
    from app.application.agents.tool_confirmation import MEMORY_WRITE_TOOLS
    if not receipts or any(r.name not in MEMORY_WRITE_TOOLS for r in receipts):return None
    lines=[]
    for receipt in receipts:
        artifact=receipt.artifact or {}
        if receipt.status=='error':
            if artifact.get('error_code') in {'invalid_input','invalid_arguments','stale_memory'}:
                return None
            lines.append('本次长期记忆操作已拒绝，未执行。' if artifact.get('error_code')=='user_rejected'
                else '本次长期记忆操作未取得成功回执，请先核对当前偏好；不会自动重试。')
        else:
            data=artifact.get('data')
            if not isinstance(data,dict) or data.get('operation')!='memory' or not data.get('guidance'):
                return None
            lines.append(data['guidance'])
    return '\n'.join(lines)
