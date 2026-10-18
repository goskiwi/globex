"""交付已完成的研究报告；事实取自本会话子任务，不发布购物卡或执行交易。"""
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from app.application.runtime.results import ToolResult, ToolResultState
from app.application.usecases.research_report import project_research_report, research_summary
from app.domain.catalog.exchange_rate import ExchangeRateTable


class ResearchDeliveryInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task_ids: list[str] = Field(min_length=1, max_length=12,
        description='本轮要交付的研究任务 scope.task_id，可同时选择多个品类；不得填商品ID')


def _report(data):
    """正文说明本次交付范围；候选取舍和已读资料由研究组件完整展示。"""
    lines = ['### ' + data['goal'], research_summary(data['status'], data['items'])]
    for key, label in (('unmet_constraints', '未满足要求'), ('questions', '需要补充'), ('issues', '未完成事项')):
        if data.get(key):
            lines.append(label + '：' + '；'.join(data[key]))
    if data.get('transaction_state') == 'awaiting_confirmation':
        lines.append('已有确认单等待页面批准，尚未执行交易。')
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
