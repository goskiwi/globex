"""按实际模型轮次限制执行；图节点数只作异常循环保护。"""
from typing import Literal
from typing_extensions import NotRequired
from dataclasses import dataclass

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, hook_config
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from app.application.runtime.errors import ExecutionStopped
from app.infrastructure.context_usage import record_evaluation_evidence
from app.infrastructure.budget import get_budget, estimate_input_tokens, resolve_model, MINIMAL_MODE_HINT
from app.application.runtime.errors import raise_if_tool_stopped
from langchain_core.utils.function_calling import convert_to_openai_tool
from opentelemetry import trace


@dataclass(frozen=True)
class ExecutionResult:
    text: str
    status: Literal["completed", "partial", "needs_input", "failed"]
    stop_reason: str | None = None
    product_delivery_complete: bool = False


class ExecutionState(AgentState):
    model_rounds: NotRequired[int]
    execution_stop: NotRequired[str | None]


class ExecutionMiddleware(AgentMiddleware):
    state_schema = ExecutionState

    def __init__(self, max_rounds: int, *, main: bool = False, budget_lite_model: str = "",
                 delivery_tools: tuple[str, ...] = ()):
        if max_rounds < 2:
            raise ValueError("模型轮数必须至少为 2，包含最终交付轮")
        self.max_rounds, self.main = max_rounds, main
        self.budget_lite_model = budget_lite_model
        self.delivery_tools = frozenset(delivery_tools)

    async def abefore_agent(self, state, runtime):
        return {"model_rounds": 0, "execution_stop": None}

    async def awrap_tool_call(self, request, handler):
        # 工具声明限制与执行检查共用角色配置；模型越过收尾工具集也不能新增研究或写入。
        if request.state.get("execution_stop") and request.tool_call["name"] not in self.delivery_tools:
            return ToolMessage(name=request.tool_call["name"], tool_call_id=request.tool_call["id"],
                status="error", content="研究阶段已结束，本调用未执行；请仅交付已有成果和缺口。",
                artifact={"executed": False, "stop_reason": request.state["execution_stop"]})
        return await handler(request)

    async def awrap_model_call(self, request, handler):
        rounds = request.state.get("model_rounds", 0)
        if rounds >= self.max_rounds:
            raise ExecutionStopped("model_call_limit")
        reason = request.state.get("execution_stop")
        try:
            raise_if_tool_stopped(request.messages)
        except ExecutionStopped as error:
            if not self.main:
                raise
            reason = error.reason
        budget = get_budget()
        if budget is not None:
            request = request.override(model_settings={**request.model_settings,
                "model": resolve_model(request.model.model_name, self.budget_lite_model)})
            if budget.tier == "minimal":
                system = request.system_message.text if request.system_message else ""
                request = request.override(system_message=SystemMessage(content=system + "\n" + MINIMAL_MODE_HINT))
        if self.main and budget is not None:
            budget.release_delivery()
            output = getattr(request.model, "max_tokens", None) or 8192
            closing_tools = [t for t in request.tools if self._tool_name(t) in self.delivery_tools]
            closing_cost = estimate_input_tokens(request.messages,
                [convert_to_openai_tool(t) for t in closing_tools]) + output
            next_cost = estimate_input_tokens(request.messages,
                [convert_to_openai_tool(tool) for tool in request.tools]) + output
            if reason is None and budget.remaining >= closing_cost + next_cost:
                budget.delivery_reservation = budget.reserve(closing_cost)
            else:
                reason = reason or "budget_exhausted"
        if reason or rounds == self.max_rounds - 1:
            reason = reason or "model_call_limit"
            if budget is not None:
                budget.release_delivery()
            tools = [t for t in request.tools if self._tool_name(t) in self.delivery_tools]
            trace.get_current_span().set_attribute("globex.execution.stop_reason", reason)
            trace.get_current_span().set_attribute("globex.execution.phase", "delivery")
            record_evaluation_evidence("execution_transition", {"phase": "delivery", "reason": reason,
                "model_rounds": rounds, "delivery_tools": [self._tool_name(t) for t in tools]})
            response = await handler(request.override(tools=tools, tool_choice="auto" if tools else "none",
                messages=[*request.messages, SystemMessage(content=(
                    "研究阶段已结束，这是本轮最终交付机会。只使用已有证据和开放的交付工具，不能继续探索。"
                    "已经筛选的候选可以交付；找不齐就说明缺口，未完成筛选则交回证据和未完成项，不能凑数或编造推荐。"))]))
            model_response = response.model_response if isinstance(response,ExtendedModelResponse) else response
            for message in model_response.result:
                if isinstance(message, AIMessage):
                    message.additional_kwargs["execution_stop"] = reason
            return response
        last = next((m for m in reversed(request.messages) if isinstance(m, AIMessage)), None)
        if rounds and last is not None and last.invalid_tool_calls:
            # 整批尚未执行：纠正本批调用，不借参数纠错重新探索或重放其他操作。
            names = {call["name"] for call in [*last.tool_calls, *last.invalid_tool_calls]}
            request = request.override(tools=[tool for tool in request.tools
                if (tool.get("name") if isinstance(tool, dict) else tool.name) in names], tool_choice="auto")
        return await handler(request)

    @staticmethod
    def _tool_name(tool):
        return tool.get("name", tool.get("function", {}).get("name")) if isinstance(tool, dict) else tool.name

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        rounds = state.get("model_rounds", 0) + 1
        update = {"model_rounds": rounds}
        if rounds == self.max_rounds:
            update["execution_stop"] = "model_call_limit"
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and message.additional_kwargs.get("execution_stop"):
            update["execution_stop"] = message.additional_kwargs["execution_stop"]
        if isinstance(message, AIMessage):
            finish = message.response_metadata.get("finish_reason")
            reason = {"length":"output_limit", "content_filter":"provider_refusal"}.get(finish)
            if reason:
                if not self.main:
                    raise ExecutionStopped(reason)
                if reason == 'provider_refusal' and not message.text:
                    message.content = str(ExecutionStopped(reason))
                return {**update, "execution_stop":reason, "jump_to":"end"}
        if isinstance(message, AIMessage) and message.invalid_tool_calls:
            # 原始非法参数保留在 AIMessage；整批不执行，不用空参数代替坏参数。
            invalid_ids = {call["id"] for call in message.invalid_tool_calls}
            receipts = []
            for call in [*message.tool_calls, *message.invalid_tool_calls]:
                detail = ("工具参数不是有效 JSON 对象，请重新生成完整参数。"
                          if call["id"] in invalid_ids else
                          "同批存在非法参数，本调用尚未执行；请按需重新提交。")
                receipts.append(ToolMessage(tool_call_id=call["id"], name=call["name"],
                    status="error", content=detail,
                    artifact={"error_code": "invalid_arguments", "executed": False}))
                record_evaluation_evidence("tool_result", {"tool":call["name"],
                    "state":"error", "result":[detail], "executed":False})
            update.update(messages=receipts, jump_to="model")
        return update


def graph_step_limit(graph, max_rounds: int) -> int:
    # 每轮最多遍历全部非终端节点一次，额外两轮容纳初始化和交付钩子。
    # 业务终止由模型轮数负责；该限制只兜住没有模型进展的错误图循环。
    return len(graph.get_graph().nodes) * (max_rounds + 2)
