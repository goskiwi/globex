"""交付已完成的研究报告；事实取自本会话子任务，不发布购物卡或执行交易。"""
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application.runtime.results import ToolResult, ToolResultState
from app.application.usecases.research_report import project_research_report, research_summary
from app.domain.catalog.exchange_rate import ExchangeRateTable


class ResearchPresentation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    task_id: str = Field(min_length=1, description='本会话已接纳研究的scope.task_id')
    title: str = Field(min_length=1, max_length=80, description='面向买家的简短研究标题，不是内部任务指令')


class ResearchDeliveryInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reports: list[ResearchPresentation] = Field(min_length=1, max_length=12)

    @model_validator(mode='after')
    def unique_tasks(self):
        if len({r.task_id for r in self.reports}) != len(self.reports):
            raise ValueError('同一研究任务只能交付一次标题')
        return self


def _report(data):
    """正文说明本次交付范围；候选取舍和已读资料由研究组件完整展示。"""
    lines = ['### ' + data['title'], research_summary(data['status'], data['items'])]
    for key, label in (('unmet_requirements', '未满足要求'), ('caveats','资料局限'), ('questions', '需要补充'), ('issues', '未完成事项')):
        if data.get(key):
            lines.append(label + '：' + '；'.join(data[key]))
    if data.get('transaction_state') == 'awaiting_confirmation':
        lines.append('已有确认单等待页面批准，尚未执行交易。')
    return '\n\n'.join(lines)


def build_research_delivery_tool(rates: ExchangeRateTable):
    async def present_research(reports: list[ResearchPresentation], runtime: ToolRuntime) -> ToolResult:
        """一次交付用户要求的各品类研究报告；只引用任务，不重写候选事实或长总结。

        用户本轮只要研究结果、暂不最终推荐或组合时使用。reports每项提供task_dispatch的
        scope.task_id与面向买家的title，可一次交付多个品类。标题由Main决定，内部goal不进入展示。
        系统填充事实和价格比较，分别展示专家的取舍建议与待核验问题，
        已交付的研究不再重写；不调用推荐/比较，不下单。
        """
        request=ResearchDeliveryInput(reports=reports)
        titles={r.task_id:r.title for r in request.reports}
        tasks = runtime.state.get('task_plan', {}).get('tasks', {})
        selected = list(titles)
        results = {}
        for message in runtime.state['messages']:
            if not isinstance(message, ToolMessage) or message.name != 'task_dispatch':
                continue
            data = (message.artifact or {}).get('data')
            scope = data.get('scope') if isinstance(data, dict) else None
            if isinstance(scope, dict) and scope.get('task_id') in selected:
                results[scope['task_id']] = data
        for key in selected:
            if key not in tasks or tasks[key].get('stale') or key not in results:
                return ToolResult('研究任务不存在、已失效或缺少本会话原始回执，请核对任务来源。',
                                  state=ToolResultState.ERROR, error_code='business_rejected')
        completed = [key for key in selected if results[key]['status'] == 'completed'
                     and not results[key].get('stop_reason')]
        from app.application.runtime.research_plan import delivery_coverage, project_delivery_outcome
        coverage=delivery_coverage(runtime.state.get('task_plan',{}),selected,completed,'research')
        projected = [project_research_report(results[key], rates, title=titles[key]) for key in selected]
        guidance='\n\n'.join(_report(report) for report in projected)
        if coverage['unmet_goals']:
            guidance+=f"\n\n还有{len(coverage['unmet_goals'])}项研究尚未完成。"
        return ToolResult({'guidance':guidance,
                           'reports': projected,
                           'task_ids': selected, 'completed_task_ids': completed,
                           'plan_outcome':project_delivery_outcome(coverage),'status':coverage['status']})
    present_research.input_model = ResearchDeliveryInput
    return present_research
