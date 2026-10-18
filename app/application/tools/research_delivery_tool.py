"""交付已完成的研究报告；事实取自本会话子任务，不发布购物卡或执行交易。"""
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from app.application.runtime.results import ToolResult, ToolResultState


class ResearchDeliveryInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task_ids: list[str] = Field(min_length=1, max_length=12,
        description='本轮要交付的研究任务 scope.task_id，可同时选择多个品类；不得填商品ID')


def _report(data):
    lines = ['### ' + data['scope']['goal'], data['summary']]
    for candidate in data.get('candidates', []):
        facts = candidate['facts']
        product, sku = facts.get('product', {}), facts.get('sku') or {}
        title = product.get('title') or candidate['product_id']
        identity = candidate.get('sku_id') or candidate['product_id']
        lines.append(f"- **{title}**（{identity}）")
        observed = []
        if sku.get('spec'):
            observed.append('规格：' + sku['spec'])
        if sku.get('price_major') is not None and sku.get('currency'):
            price = str(sku['price_major']).removesuffix('.0')
            observed.append(f"商品单价 {price} {sku['currency']}")
        if product.get('weight_kg') is not None:
            observed.append(f"重量 {product['weight_kg']} kg")
        if observed:
            lines.append('  已读事实：' + '；'.join(observed) + '。')
        lines.append('  取舍判断：' + candidate['reason'])
        if candidate.get('unmet_constraints'):
            lines.append('  未满足要求：' + '；'.join(candidate['unmet_constraints']))
    for key, label in (('unmet_constraints', '未满足要求'), ('questions', '需要补充'),
                       ('issues', '未完成事项'), ('unknowns', '资料缺口')):
        if data.get(key):
            lines.append(label + '：' + '；'.join(data[key]))
    if data.get('transaction_state') == 'awaiting_confirmation':
        lines.append('已有确认单等待页面批准，尚未执行交易。')
    lines.append('以上为本次已读研究资料，后续购买仍需核验价格、库存和具体规格。')
    return '\n\n'.join(lines)


def build_research_delivery_tool():
    async def present_research(task_ids: list[str], runtime: ToolRuntime) -> ToolResult:
        """一次交付用户要求的各品类研究报告；只引用任务，不重写候选事实或长总结。

        用户本轮只要研究结果、暂不最终推荐或组合时使用。task_ids 来自 task_dispatch 的
        scope.task_id，可在一次调用中包含多份研究。系统填充名称、SKU、已读事实、取舍和缺口，
        成功后直接结束本轮；不调用推荐/比较，不发布购物卡，不下单。
        """
        tasks = runtime.state.get('task_plan', {}).get('tasks', {})
        selected = list(dict.fromkeys(task_ids))
        reports = {}
        for message in runtime.state['messages']:
            if not isinstance(message, ToolMessage) or message.name != 'task_dispatch':
                continue
            data = (message.artifact or {}).get('data')
            scope = data.get('scope') if isinstance(data, dict) else None
            if isinstance(scope, dict) and scope.get('task_id') in selected:
                reports[scope['task_id']] = data
        for key in selected:
            if key not in tasks or tasks[key].get('stale') or key not in reports:
                return ToolResult('研究任务不存在、已失效或缺少本会话原始回执，请核对任务来源。',
                                  state=ToolResultState.ERROR, error_code='business_rejected')
        completed = [key for key in selected if reports[key]['status'] == 'completed'
                     and not reports[key].get('stop_reason')]
        return ToolResult({'guidance': '\n\n'.join(_report(reports[key]) for key in selected),
                           'task_ids': selected, 'completed_task_ids': completed,
                           'status': 'completed' if len(completed) == len(selected) else 'partial'})
    present_research.input_model = ResearchDeliveryInput
    return present_research
