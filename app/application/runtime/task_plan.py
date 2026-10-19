"""现有图状态中的研究计划：定义来自买家需求，进度与候选来源来自工具回执。"""
from dataclasses import replace
from copy import deepcopy
from typing_extensions import NotRequired
import json
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from app.application.agents.shopping_state import ShoppingWork, compile_search
from app.domain.buyer.preference import BuyerPreference
from app.application.runtime.request_scope import ShoppingContext

from app.application.runtime.research_plan import (
    ResearchPlan, project_plan, dependent_steps, record_research, apply_delivery,
)

from app.application.usecases.intents import DELIVERY_KINDS

DELIVERIES = frozenset(DELIVERY_KINDS)




class TaskPlanState(AgentState):
    task_plan: NotRequired[ResearchPlan]


class TaskPlanMiddleware(AgentMiddleware):
    state_schema = TaskPlanState

    async def abefore_model(self, state, runtime):
        work = ShoppingWork.model_validate(state['shopping_work'])
        facts = tuple(BuyerPreference(**p) for p in state['preference_snapshot']['facts'])
        policy = compile_search(work, facts, ShoppingContext.current().currency)
        plan = project_plan(work, policy, state.get('task_plan', {}))
        last = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
        if last and last.tool_calls and last.id != plan.get('last_batch'):
            calls = {c['id'] for c in last.tool_calls}
            receipts = [m for m in state['messages']
                        if isinstance(m, ToolMessage) and m.tool_call_id in calls]
            for receipt in receipts:
                data = (receipt.artifact or {}).get('data')
                if receipt.name == 'task_dispatch' and isinstance(data, dict) and data.get('scope'):
                    record_research(plan, work, data)
            plan = project_plan(work, policy, plan)
            # 工具可以并行产生独立成果，计划只在整批回执到齐后写一次。
            for receipt in receipts:
                data = (receipt.artifact or {}).get('data')
                if receipt.status != 'success' or not isinstance(data, dict):
                    continue
                if receipt.name == 'present_research':
                    plan = apply_delivery(plan, data['completed_task_ids'], 'research', work, policy)
                elif receipt.name in {'recommend_products', 'compare_products'}:
                    plan = apply_delivery(plan, data['plan_outcome']['delivered_tasks'], 'products', work, policy)
            plan['last_batch'] = last.id
        plan = project_plan(work, policy, plan)
        # 无计划且只做直接查询时，不额外制造计划步骤。
        hint = {'steps': plan['steps'], 'research': {key: {
            'goal': value['scope']['goal'],
            **{k: value[k] for k in ('status', 'execution_status', 'qualified_skus', 'unmet_requirements', 'caveats', 'stale', 'delivery') if k in value}}
            for key, value in plan['tasks'].items()}}
        result = {'task_plan': plan}
        if plan['steps'] or plan['tasks'] or any(m.name == 'task_plan' for m in state['messages']):
            result['messages'] = [SystemMessage(id='current-task-plan', name='task_plan',
                content='当前任务进度由实际证据产生；verified表示候选已核验，不代表整体交付。\n'
                        '仅交付研究时用 present_research 一次引用各 task_id；研究报告不等于发布购物卡。\n'
                        + json.dumps(hint, ensure_ascii=False))]
        return result

    async def awrap_tool_call(self, request, handler):
        context = ShoppingContext.current()
        plan = request.state.get('task_plan', {})
        last = next((m for m in reversed(request.state['messages']) if isinstance(m, AIMessage)), None)
        names = {c['name'] for c in last.tool_calls} if last else set()
        calls = last.tool_calls if last else []
        step_ids = [c['args']['task'].get('step_id') for c in calls
                    if c['name'] == 'task_dispatch' and isinstance(c['args'].get('task'), dict)
                    and c['args']['task'].get('step_id') is not None]
        work = ShoppingWork.model_validate(request.state['shopping_work'])
        if any(dependent_steps(work.plan, {key}).intersection(step_ids) for key in step_ids):
            return ToolMessage(name=request.tool_call['name'], tool_call_id=request.tool_call['id'], status='error',
                               content='存在前后依赖的研究不能同批执行；先完成前置研究，再执行后续步骤。本批均未执行。')
        if len(step_ids) != len(set(step_ids)):
            return ToolMessage(name=request.tool_call['name'], tool_call_id=request.tool_call['id'], status='error',
                               content='同一批次不能重复执行一个计划步骤；本批均未执行。')
        if 'task_dispatch' in names and names & DELIVERIES:
            return ToolMessage(name=request.tool_call['name'], tool_call_id=request.tool_call['id'], status='error',
                               content='研究任务和最终交付必须分批执行，本批均未执行。')
        token = ShoppingContext.set(replace(context, task_plan=plan))
        try:
            return await handler(request)
        finally:
            ShoppingContext.reset(token)
