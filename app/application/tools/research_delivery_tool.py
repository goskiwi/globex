"""交付已完成的研究报告；事实取自本会话子任务，不发布购物卡或执行交易。"""
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from app.application.runtime.results import ToolResult, ToolResultState
from app.application.usecases.research_report import project_research_report, research_summary
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
import json


class ResearchDeliveryInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task_ids: list[str] = Field(min_length=1, max_length=12,
        description='本轮要交付的研究任务 scope.task_id，可同时选择多个品类；不得填商品ID')


def _report(data):
    lines = ['### ' + data['goal'], research_summary(data['status'], data['items'])]
    for candidate in data['items']:
        facts = candidate['facts']
        product, sku = facts.get('product', {}), facts.get('sku') or {}
        title = product.get('title') or candidate['product_id']
        identity = candidate.get('sku_id') or candidate['product_id']
        lines.append(f"- **{title}**（{identity}）")
        if facts.get('historical'):
            lines.append('  历史资料：以下价格、库存和属性来自历史读取，尚未重新核验。')
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
        lines.append('  取舍建议：' + candidate['assessment'])
        for evidence in candidate['evidence']:
            value = evidence['value']
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            lines.append('  ' + evidence['label'] + '：' + value)
        if candidate.get('open_questions'):
            lines.append('  待核验问题：' + '；'.join(candidate['open_questions']))
        if candidate.get('unmet_constraints'):
            lines.append('  未满足要求：' + '；'.join(candidate['unmet_constraints']))
    for comparison in data['price_comparisons']:
        currency = comparison['currency']
        money = lambda key: str(Money.of(comparison[key], currency))
        lowest = '、'.join(row['sku_id'] for row in comparison['lowest'])
        highest = '、'.join(row['sku_id'] for row in comparison['highest'])
        basis = ('单件商品价，按当前业务汇率折算，不含运税' if comparison['kind'] == 'unit_price'
                 else f"到手报价，每个规格{comparison['quantity']}件、配送至{comparison['ship_to']}")
        lines.append(f"本报告{comparison['compared_sku_count']}个规格的同口径价格比较（{basis}）："
                     f"最低 {money('minimum_minor')}（{lowest}）；最高 {money('maximum_minor')}（{highest}）；"
                     f"差额 {money('spread_minor')}。")
    for key, label in (('unmet_constraints', '未满足要求'), ('questions', '需要补充'),
                       ('issues', '未完成事项')):
        if data.get(key):
            lines.append(label + '：' + '；'.join(data[key]))
    if data.get('transaction_state') == 'awaiting_confirmation':
        lines.append('已有确认单等待页面批准，尚未执行交易。')
    lines.append('以上为本次已读研究资料，后续购买仍需核验价格、库存和具体规格。')
    return '\n\n'.join(lines)


def build_research_delivery_tool(rates: ExchangeRateTable):
    async def present_research(task_ids: list[str], runtime: ToolRuntime) -> ToolResult:
        """一次交付用户要求的各品类研究报告；只引用任务，不重写候选事实或长总结。

        用户本轮只要研究结果、暂不最终推荐或组合时使用。task_ids 来自 task_dispatch 的
        scope.task_id，可在一次调用中包含多份研究。系统填充事实和价格比较，分别展示专家的取舍建议与待核验问题，
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
        projected = [project_research_report(reports[key], rates) for key in selected]
        return ToolResult({'guidance': '\n\n'.join(_report(report) for report in projected),
                           'reports': projected,
                           'task_ids': selected, 'completed_task_ids': completed,
                           'status': 'completed' if len(completed) == len(selected) else 'partial'})
    present_research.input_model = ResearchDeliveryInput
    return present_research
