"""以工具形式派发独立子图：结构化任务、服务端上下文、经证据核对的交付结果。"""
import asyncio
import json
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from app.application.agents.handoff import DelegatedTask, SubagentResult
from app.application.agents.shopping_state import ShoppingWork, Filters, compile_search
from app.application.runtime.results import ToolResult, ToolResultState
from app.application.runtime.handoff import TaskEvidence, HandoffContext, issue
from app.application.runtime.errors import ExecutionStopped, ContextCapacityError
from app.infrastructure.resilience import DEFAULT_TIMEOUTS
from app.infrastructure.budget import get_budget
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import observe_run_events
from app.infrastructure.context_products import token_estimate
from app.application.runtime.tool_view import bounded_tool_view
from opentelemetry import trace


def _parent_context(runtime):
    """只从原生运行时读取当前买家输入和已有工作状态，不接受模型自报身份。"""
    context = ShoppingContext.current()
    state = runtime.state if runtime is not None else {}
    if not context:
        return {}
    if state.get("shopping_work_owner", context.buyer_id) != context.buyer_id:
        raise ValueError("购物状态不属于当前买家")
    buyers = [m for m in state.get("messages", [])
              if isinstance(m, HumanMessage) and m.name == context.buyer_id]
    return {
        "latest_user_request": buyers[-1].text if buyers else "",
        "shopping_work": deepcopy(state.get("shopping_work") or ShoppingWork(filters=Filters()).model_dump()),
        "locale": context.locale,
        "currency": context.currency,
        "effective_search": deepcopy(context.effective_search),
    }


def build_task_dispatch_tool(search_factory, trade_factory, bus, *, model_token_limit=12000):
    async def task_dispatch(
        subagent_type: Literal["search_agent", "trade_agent"],
        task: DelegatedTask,
        runtime: ToolRuntime = None,
    ) -> ToolResult:
        """派发独立专家任务，返回结构化成果、缺口及业务确认状态。

        简单单步操作直接调用业务工具。需要多步研究或交易准备时再派发；
        当前买家原文和购物状态由运行时注入，模型不得自报身份或授权。

        Args:
            subagent_type: search_agent 负责商品研究，trade_agent 负责已选商品的交易准备。
            task: 目标、仅本品类的条件差异、待核验要求及必要上下文。候选输出固定为 ID 与简短理由，不重复描述交付格式。
            runtime: LangGraph 自动注入，不属于模型参数。
        """
        task = DelegatedTask.model_validate(task)
        session_id = ShoppingContext.current_session_id()
        call_id = runtime.tool_call_id if runtime is not None else None
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        evidence = TaskEvidence(session_id)
        feedback = []
        stop_reason = None
        try:
            budget = get_budget()
            if budget is not None and budget.exhausted:
                raise ExecutionStopped("budget_exhausted")
            else:
                parent = _parent_context(runtime)
                factory = {"search_agent": search_factory, "trade_agent": trade_factory}[subagent_type]
                context = ShoppingContext.current()
                if context is None:
                    raise ValueError("派发缺少当前买家上下文")
                scoped = ShoppingWork.model_validate(parent.pop("shopping_work"))
                if subagent_type == "trade_agent" and task.filters is not None:
                    raise ValueError("交易任务不能覆盖购物条件，请先由 Main 更新状态")
                if task.filters is not None:
                    changes = task.filters.model_dump(exclude_unset=True)
                    for field in ("excluded_material_tags", "required_material_tags"):
                        if field in changes:
                            changes[field] = list(dict.fromkeys([*getattr(scoped.filters, field), *changes[field]]))
                    scoped.filters = Filters.model_validate({**scoped.filters.model_dump(), **changes})
                if task.requirements is not None:
                    scoped.unverified_requirements = list(dict.fromkeys([*scoped.unverified_requirements, *task.requirements]))
                if task.preferences is not None:
                    scoped.preferences = task.preferences
                effective = compile_search(scoped, context.preference_facts, context.currency)
                parent["effective_search"] = effective
                parent["selected_products"] = list(context.selected_lines)
                # 子任务只看到解析后的一份条件，不再同时接收原条件和覆盖值。
                task_context = task.model_dump(exclude={"filters", "requirements", "preferences"})
                inputs = [HumanMessage(name="delegated_task", content=json.dumps({
                    "delegated_task": task_context, "parent_context": parent}, ensure_ascii=False))]
                bus.publish(session_id, "agent.dispatch", {"agent": subagent_type,
                            "tool_call_id": call_id, "started_at": started_at})
                with observe_run_events(evidence.capture):
                    token = None
                    if effective is not None:
                        token = ShoppingContext.set(replace(context, effective_search=effective))
                    try:
                        # 子任务边界持有时限，超时后仍能交回该任务已收集的证据。
                        async with asyncio.timeout(DEFAULT_TIMEOUTS["task_dispatch"]):
                            reply = await factory.build().ainvoke(
                                {"messages": inputs, "handoff_task": task.model_dump()},
                                context=HandoffContext(evidence))
                    finally:
                        if token is not None:
                            ShoppingContext.reset(token)
                feedback = reply.get("handoff_feedback") or []
                if reply.get("handoff_result") is None:
                    feedback = [issue("result_missing", "result", "子图没有交付最终结果。")]
                    result = SubagentResult(status="failed", summary="子图未完成交付。", issues=["result_missing"])
                else:
                    result = SubagentResult.model_validate(reply["handoff_result"])
                stop_reason = reply.get("execution_stop")
        except (ExecutionStopped, GraphRecursionError, TimeoutError, ContextCapacityError) as error:
            stop_reason = error.reason if isinstance(error, ExecutionStopped) else (
                "step_limit" if isinstance(error, GraphRecursionError) else
                "context_capacity" if isinstance(error, ContextCapacityError) else "timeout")
            result = evidence.stopped_result(stop_reason)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # 保留已完成的业务引用，不回传可能包含地址或模型请求的异常原文。
            feedback = [issue("execution_error", "runtime", "子图执行异常，已完成业务结果保留；请核对后决定是否重试。")]
            result = SubagentResult(status="failed", summary="子任务未交付可核验的完整结果。",
                                    issues=["execution_error:" + type(error).__name__])

        decision = {"agent": subagent_type, **result.model_dump(),
                    "evidence_refs": sorted(evidence.refs), "feedback": feedback,
                    "historical_evidence_refs": sorted(evidence.historical_refs),
                    "trade_results": evidence.trade, "transaction_state": "none",
                    "stop_reason": stop_reason}
        if any(item.get("confirmation_id") and item["status"] == "pending" and not item["expired"]
               for item in evidence.trade):
            decision["transaction_state"] = "awaiting_confirmation"
            decision["summary"] = (result.summary + " " if stop_reason else "") + "已准备交易确认单，尚未执行交易；请买家核对页面确认卡并明确批准或拒绝。"
        view = await bounded_tool_view(decision, getattr(search_factory, 'evidence_store', None),
            ShoppingContext.current(), kind='handoff', token_limit=model_token_limit)
        span = trace.get_current_span()
        for key, value in {"candidate_count": len(result.candidates),
            "observed_product_count": len(evidence.products), "reference_count": len(evidence.refs),
            "model_tokens": token_estimate(view), "full_tokens": token_estimate(decision)}.items():
            span.set_attribute("globex.handoff." + key, value)
        if stop_reason:
            span.set_attribute("globex.execution.stop_reason", stop_reason)
        return ToolResult(decision, model_data=view,
            state=ToolResultState.ERROR if result.status == "failed" else ToolResultState.SUCCESS,
            error_code="business_rejected" if result.status == "failed" else None)

    return task_dispatch
