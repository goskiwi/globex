"""研究计划的业务状态与转换；不依赖图钩子或请求上下文。"""
from copy import deepcopy
from typing import Literal, TypedDict
from typing_extensions import NotRequired

from app.application.agents.shopping_state import ShoppingWork


class TaskScope(TypedDict):
    task_id: str
    step_id: str | None
    goal: str
    definition: dict | None
    parent_conditions: dict
    effective_search: dict


class ResearchTask(TypedDict):
    scope: TaskScope
    status: Literal["completed", "partial", "needs_input", "failed"]
    qualified_skus: list[str]
    observed_products: list[str]
    observed_skus: dict[str, list[str]]
    evidence_refs: list[str]
    required_gaps: list[str]
    gaps: list[str]
    stale: NotRequired[bool]
    delivered: NotRequired[bool]
    reported: NotRequired[bool]


class ResearchPlan(TypedDict, total=False):
    steps: list[dict]
    tasks: dict[str, ResearchTask]
    last_batch: str | None
    status: str


def condition_identity(policy):
    return {k: policy.get(k) for k in ('parameters', 'unverified_requirements', 'excluded_products', 'excluded_skus')}


def dependent_steps(steps, changed):
    """计划已按依赖排序；返回直接及间接下游，不改变独立分支。"""
    affected = set(changed)
    for step in steps:
        if affected.intersection(step.depends_on):
            affected.add(step.id)
    return affected - set(changed)


def project_plan(work: ShoppingWork, policy: dict, saved: ResearchPlan) -> ResearchPlan:
    tasks = deepcopy(saved.get('tasks', {}))
    definitions = {s.id: s.model_dump() for s in work.plan}
    for value in tasks.values():
        scope = value['scope']
        value['stale'] = (value.get('stale', False) or scope['parent_conditions'] != condition_identity(policy)
                          or (scope.get('step_id') is not None
                              and definitions.get(scope['step_id']) != scope.get('definition')))
    steps = []
    statuses = {}
    for step in work.plan:
        value = tasks.get(step.id)
        if value and any(statuses[key] not in {'verified', 'delivered'} for key in step.depends_on):
            value['stale'] = True
        status = ('pending' if not value or value['stale'] else
                  'delivered' if value.get('delivered') else
                  'verified' if value['status'] == 'completed' and value.get('qualified_skus') and not value.get('required_gaps') else 'blocked')
        steps.append({**step.model_dump(), 'status': status,
                      'gaps': (value.get('gaps', []) if value and not value['stale'] else []),
                      'evidence_refs': (value.get('evidence_refs', []) if value and not value['stale'] else [])})
        statuses[step.id] = status
    return {'steps': steps, 'tasks': tasks, 'last_batch': saved.get('last_batch')}


def resolve_candidate_scope(plan: ResearchPlan, pick, *, require_qualified=True):
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


def plan_delivery(plan: ResearchPlan, picks, cards):
    requested = {p.task_id for p in picks if p.task_id is not None}
    tasks = plan.get('tasks', {})
    issues = {c['default_sku_id']: c['constraint_issues'] for c in cards}
    delivered = {key for key in requested if tasks[key]['status'] == 'completed' and not tasks[key].get('required_gaps')
                 and all(not issues[p.sku_id] for p in picks if p.task_id == key)}
    steps = deepcopy(plan.get('steps', []))
    for step in steps:
        if step['id'] in delivered and step['status'] in {'verified', 'delivered'}:
            step['status'] = 'delivered'
    missing = list(dict.fromkeys([s['goal'] for s in steps if s['status'] != 'delivered']
        + [tasks[key]['scope']['goal'] for key in requested - delivered]))
    return {'status': 'partial' if missing else 'completed', 'steps': steps,
            'delivered_tasks': sorted(delivered), 'unmet_goals': missing}


def record_research(plan: ResearchPlan, work: ShoppingWork, data: dict) -> None:
    """只修改本轮复制的计划；新的前置回执使下游失效。"""
    scope = data['scope']
    for key in dependent_steps(work.plan, {scope['task_id']}):
        if key in plan['tasks']:
            plan['tasks'][key]['stale'] = True
    plan['tasks'][scope['task_id']] = {
        'scope': scope, 'status': data['status'],
        'qualified_skus': data.get('qualified_skus', []),
        'observed_products': data.get('observed_products', []),
        'observed_skus': data.get('observed_skus', {}),
        'evidence_refs': data.get('evidence_refs', []),
        'required_gaps': [*data.get('unmet_constraints', []), *scope['effective_search'].get('unverified_requirements', [])],
        'gaps': [*data.get('unmet_constraints', []), *data.get('questions', []),
                 *data.get('issues', []), *data.get('unknowns', [])],
    }


def apply_delivery(plan: ResearchPlan, outcome: dict) -> ResearchPlan:
    updated = deepcopy(plan)
    for key in outcome['delivered_tasks']:
        updated['tasks'][key]['delivered'] = True
    updated['steps'] = outcome['steps']
    updated['status'] = outcome['status']
    return updated
