"""逐调用保留已接纳的展示；本轮收口独立于单项成果的保存。"""
from typing_extensions import Annotated,NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware,hook_config
from langchain_core.messages import AIMessage,ToolMessage
from langgraph.types import Overwrite
from app.application.usecases.intents import Presentation,DELIVERY_KINDS


def merge_presentations(current,incoming):
    records={p['id']:p for p in current}
    records.update({p['id']:p for p in incoming})
    return list(records.values())


class DeliveryState(AgentState):
    presentations: NotRequired[Annotated[list[Presentation],merge_presentations]]
    operation_receipts: NotRequired[Annotated[list[dict],merge_presentations]]


def confirmation_delivery(payload, identifier):
    """仅接纳服务端返回、属于当前身份的有效待确认操作；不执行用户决议。"""
    from app.application.runtime.request_scope import ShoppingContext
    context = ShoppingContext.current()
    card = payload['confirmation']
    if (context is None or card['buyer_id'] != context.buyer_id
            or card['session_id'] != context.shopping_session_id
            or card['status'] != 'pending' or card.get('expired', False)):
        raise ValueError('待确认结果与当前买家、会话或状态不符')
    captions = {'create': '订单确认卡已准备，请核对卡片后操作；确认前不会创建订单或扣减库存。',
                'cancel': '取消确认卡已准备，请在卡片上操作；确认前订单保持原状态。'}
    return {'id': identifier, 'runId': context.source_run_id, 'confirmation_id': card['confirmation_id'],
            'guidance': captions[card['action']], 'action': card['action']}


def presentation(name,payload,identifier):
    """仅在原生工具回执通过校验后生成；模型不能自报是否已经交付。"""
    kind=DELIVERY_KINDS[name]
    if not isinstance(payload,dict):raise ValueError('展示结果必须为结构化记录')
    if kind=='clarification':
        if not payload.get('form_id') or not isinstance(payload.get('form'),dict):
            raise ValueError('澄清结果缺少持久表单')
        status='needs_input'
    else:
        if not isinstance(payload.get('guidance'),str) or not payload['guidance'].strip():
            raise ValueError('展示结果缺少说明')
        if kind=='research' and (not payload.get('reports') or not payload.get('task_ids')):
            raise ValueError('研究交付缺少报告或任务来源')
        status=payload.get('status','completed') if kind=='research' else payload.get('plan_outcome',{}).get('status','completed')
    from app.application.runtime.request_scope import ShoppingContext
    context=ShoppingContext.current()
    return {'id':identifier,'runId':context.source_run_id if context else '', 'kind':kind,'status':status,'payload':payload}


class FinalDeliveryMiddleware(AgentMiddleware):
    state_schema=DeliveryState

    async def abefore_agent(self,state,runtime):
        from app.application.runtime.request_scope import ShoppingContext
        context=ShoppingContext.current()
        current=[p for p in state.get('presentations',[]) if context and context.source_run_id and p['runId']==context.source_run_id]
        return {'presentations':Overwrite(current), 'operation_receipts': Overwrite([])}

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self,state,runtime):
        receipts=[]
        for message in reversed(state['messages']):
            if not isinstance(message,ToolMessage):break
            receipts.append(message)
        if not receipts or any(r.status!='success' or (r.artifact or {}).get('notices') for r in receipts):return None
        index=len(state['messages'])-len(receipts)-1
        call=state['messages'][index] if index>=0 else None
        if not isinstance(call,AIMessage) or {c['id'] for c in call.tool_calls}!={r.tool_call_id for r in receipts}:return None
        accepted={p['id']:p for p in state.get('presentations',[])}
        operations={p['id']:p for p in state.get('operation_receipts',[])}
        accepted.update(operations)
        values=[accepted.get((r.artifact or {}).get('execution_id')) for r in reversed(receipts)]
        if not all(values):return None
        lines=[(p['guidance'] if p['id'] in operations else
                '请填写下方的问题，提交后我会继续选购。' if p['kind']=='clarification' else p['payload']['guidance']) for p in values]
        from app.application.runtime.request_scope import ShoppingContext
        from app.application.runtime.research_plan import pending_plan_goals
        context=ShoppingContext.current()
        request_id=(context.source_run_id or context.source_message_id) if context else ''
        remaining=pending_plan_goals(state.get('task_plan',{}),request_id)
        if remaining and any(p['id'] in operations for p in values):
            return None
        if remaining:
            lines.append(f'还有{len(remaining)}项计划任务尚未完成。')
        return {'messages':[AIMessage(content='\n\n'.join(lines))],'jump_to':'end'}
