"""图内有界循环窗口；新用户消息重置，interrupt 续接沿用 checkpoint。"""
from typing_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, RemoveMessage


class LoopState(AgentState):
    loop_state: NotRequired[dict]


class LoopMiddleware(AgentMiddleware):
    state_schema = LoopState

    def __init__(self, detector):
        self.detector = detector

    async def abefore_agent(self, state, runtime):
        # 新 ainvoke 开始新任务；Command(resume=...) 从中断节点续接，不经过此入口。
        return {'loop_state': {'history': [], 'count': 0, 'stop': False, 'last_batch': None},
                'messages': [RemoveMessage(id=m.id) for m in state['messages'] if m.name == 'loop_feedback']}

    async def abefore_model(self, state, runtime):
        previous = state.get('loop_state', {})
        last = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
        if last is None or not last.tool_calls or previous.get('last_batch') == last.id:
            return None
        calls = {c['id'] for c in last.tool_calls}
        receipts = {m.tool_call_id: m for m in state['messages']
                    if isinstance(m, ToolMessage) and m.tool_call_id in calls}
        observations = [(receipts[c['id']].artifact or {}).get('loop_observation')
                        for c in last.tool_calls if c['id'] in receipts]
        updated, hint = self.detector.advance(previous, [x for x in observations if x is not None])
        updated['last_batch'] = last.id
        result = {'loop_state': updated}
        if hint:
            from app.infrastructure.context_usage import record_evaluation_evidence
            record_evaluation_evidence('tool_notice', {'reason': 'repeated_path', 'history': updated['history']})
            result['messages'] = [SystemMessage(name='loop_feedback', content=hint)]
        return result

    async def aafter_agent(self, state, runtime):
        return {'loop_state': {'history': [], 'count': 0, 'stop': False, 'last_batch': None}}
