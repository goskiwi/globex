"""现有图状态中的研究计划：定义来自买家需求，进度与候选来源来自工具回执。"""
from copy import deepcopy
from dataclasses import replace
from typing_extensions import NotRequired
import json
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from app.application.agents.shopping_state import ShoppingWork, compile_search
from app.domain.buyer.preference import BuyerPreference
from app.infrastructure.context import ShoppingContext
from app.application.runtime.results import tool_receipts

DELIVERIES = {'recommend_products', 'compare_products'}


def condition_identity(policy):
    return {k: policy.get(k) for k in ('parameters', 'unverified_requirements', 'excluded_products', 'excluded_skus')}


def project_plan(work, policy, saved):
    tasks = deepcopy(saved.get('tasks', {}))
    definitions = {s.id: s.model_dump() for s in work.plan}
    for value in tasks.values():
        scope = value['scope']
        value['stale'] = (scope['parent_conditions'] != condition_identity(policy)
                          or (scope.get('step_id') is not None
                              and definitions.get(scope['step_id']) != scope.get('definition')))
    steps = []
    for step in work.plan:
        value = tasks.get(step.id)
        status = ('pending' if not value or value['stale'] else
                  'delivered' if value.get('delivered') else
                  'verified' if value['status'] == 'completed' and value.get('qualified_skus') and not value.get('required_gaps') else 'blocked')
        steps.append({**step.model_dump(), 'status': status,
                      'gaps': (value.get('gaps', []) if value and not value['stale'] else []),
                      'evidence_refs': (value.get('evidence_refs', []) if value and not value['stale'] else [])})
    return {'steps': steps, 'tasks': tasks, 'last_batch': saved.get('last_batch')}


def resolve_candidate_scope(context, pick, *, require_qualified=True):
    plan = context.task_plan
    tasks = {k: v for k, v in plan.get('tasks', {}).items() if not v.get('stale')}
    related = [key for key, value in tasks.items() if pick.product_id in value.get('observed_products', [])]
    if pick.task_id is None:
        if related or plan.get('steps'):
            raise ValueError('研究候选必须填写对应 task_id，不能丢弃子任务或计划条件')
        return None
    value = tasks.get(pick.task_id)
    if value is None:
        raise ValueError('任务来源不存在或条件已变化，请重新核验该任务')
    if pick.sku_id not in value.get('observed_skus', {}).get(pick.product_id, []):
        raise ValueError('指定任务没有读取这个商品规格，不能借用其他商品的证据')
    if require_qualified and pick.sku_id not in value.get('qualified_skus', []):
        raise ValueError('该规格尚未在指定任务中核验为合格候选')
    return value['scope']['effective_search']


def plan_delivery(context, picks, cards):
    requested = {p.task_id for p in picks if p.task_id is not None}
    tasks = context.task_plan.get('tasks', {})
    issues = {c['default_sku_id']: c['constraint_issues'] for c in cards}
    delivered = {key for key in requested if tasks[key]['status'] == 'completed' and not tasks[key].get('required_gaps')
                 and all(not issues[p.sku_id] for p in picks if p.task_id == key)}
    steps = deepcopy(context.task_plan.get('steps', []))
    for step in steps:
        if step['id'] in delivered and step['status'] in {'verified', 'delivered'}:
            step['status'] = 'delivered'
    missing = list(dict.fromkeys([s['goal'] for s in steps if s['status'] != 'delivered']
        + [tasks[key]['scope']['goal'] for key in requested - delivered]))
    return {'status': 'partial' if missing else 'completed', 'steps': steps,
            'delivered_tasks': sorted(delivered), 'unmet_goals': missing}


class TaskPlanState(AgentState):
    task_plan: NotRequired[dict]


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
            for receipt in state['messages']:
                if not isinstance(receipt, ToolMessage) or receipt.tool_call_id not in calls or receipt.name != 'task_dispatch':
                    continue
                data = (receipt.artifact or {}).get('data')
                if not isinstance(data, dict) or not data.get('scope'):
                    continue
                scope = data['scope']
                plan['tasks'][scope['task_id']] = {'scope': scope, 'status': data['status'],
                    'qualified_skus': data.get('qualified_skus', []),
                    'observed_products': data.get('observed_products', []),
                    'observed_skus': data.get('observed_skus', {}),
                    'evidence_refs': data.get('evidence_refs', []),
                    'required_gaps': [*data.get('unmet_constraints', []), *scope['effective_search'].get('unverified_requirements', [])],
                    'gaps': [*data.get('unmet_constraints', []), *data.get('questions', []),
                             *data.get('issues', []), *data.get('unknowns', [])]}
            plan['last_batch'] = last.id
        plan = project_plan(work, policy, plan)
        # 无计划且只做直接查询时，不额外制造计划步骤。
        hint = {'steps': plan['steps'], 'research': {key: {k:v for k,v in value.items() if k != 'scope'}
                                                     for key,value in plan['tasks'].items()}}
        result = {'task_plan': plan}
        if plan['steps'] or plan['tasks'] or any(m.name == 'task_plan' for m in state['messages']):
            result['messages'] = [SystemMessage(id='current-task-plan', name='task_plan',
                content='当前任务进度由实际证据产生；verified表示候选已核验，不代表整体交付。\n'
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
        if len(step_ids) != len(set(step_ids)) or sum(c['name'] in DELIVERIES for c in calls) > 1:
            return ToolMessage(name=request.tool_call['name'], tool_call_id=request.tool_call['id'], status='error',
                               content='同一批次不能重复执行一个计划步骤，也不能提交多个最终交付；本批均未执行。')
        if 'task_dispatch' in names and names & DELIVERIES:
            return ToolMessage(name=request.tool_call['name'], tool_call_id=request.tool_call['id'], status='error',
                               content='研究任务和最终交付必须分批执行，本批均未执行。')
        token = ShoppingContext.set(replace(context, task_plan=plan))
        try:
            output = await handler(request)
            if request.tool_call['name'] in DELIVERIES:
                for receipt in tool_receipts(output):
                    data = (receipt.artifact or {}).get('data', {})
                    if receipt.status == 'success' and isinstance(data, dict) and data.get('plan_outcome'):
                        updated = deepcopy(plan)
                        for key in data['plan_outcome']['delivered_tasks']:
                            updated['tasks'][key]['delivered'] = True
                        updated['steps'] = data['plan_outcome']['steps']
                        updated['status'] = data['plan_outcome']['status']
                        if isinstance(output, Command):
                            return Command(update={**output.update, 'task_plan': updated})
                        return Command(update={'messages': [output], 'task_plan': updated})
            return output
        finally:
            ShoppingContext.reset(token)
