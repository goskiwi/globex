"""图执行边界：模型流观察、预算停止、未完成调用收口与结果投影。"""
import asyncio

from opentelemetry import trace
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END
from langgraph.errors import GraphRecursionError

from app.application.runtime.events import ReplyStart
from app.application.runtime.errors import ContextCapacityError, ExecutionStopped
from app.application.runtime.handoff import TaskEvidence
from app.application.runtime.execution import ExecutionResult
from app.infrastructure.budget import get_budget
from app.infrastructure.eventbus import observe_run_events

def _message_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))


class _GraphStreamObserver(AsyncCallbackHandler):
    """仅发布运行进度；模型原文留在内部，最终审核后统一交付。"""
    def __init__(self, observer):
        self.observer = observer
        self.interrupted = False

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        if self.observer:
            self.observer(ReplyStart())

    async def on_llm_end(self, response, *, run_id, **kwargs):
        self.interrupted = self.interrupted or any(
            (generation.generation_info or {}).get("finish_reason") == "interrupted"
            for group in response.generations for generation in group
        )


async def execute_graph(session_id: str, session, graph_input, event_observer=None) -> ExecutionResult:
    evidence = TaskEvidence(session_id)
    observer = _GraphStreamObserver(event_observer)
    config = {**session.config, "callbacks": [observer]}
    try:
        with trace.get_tracer(__name__).start_as_current_span(
            "globex.agent", attributes={"gen_ai.operation.name": "invoke_agent",
                                       "gen_ai.agent.name": "commerce_concierge"},
            record_exception=False, set_status_on_exception=False,
        ):
            with observe_run_events(evidence.capture):
                result = await session.graph.ainvoke(graph_input, config=config)
            if observer.interrupted:
                raise asyncio.CancelledError()
    except (ExecutionStopped, GraphRecursionError, ContextCapacityError) as error:
        reason = error.reason if isinstance(error, ExecutionStopped) else (
            "context_capacity" if isinstance(error, ContextCapacityError) else "step_limit")
        text = evidence.stopped_text(reason)
        # 原生 END 更新提交已完成并行节点的 pending writes，并清除未运行任务。
        # 不调用模型补总结，也不 resume 任何写工具。
        await session.graph.aupdate_state(session.config, None, as_node=END)
        snapshot = await session.graph.aget_state(session.config)
        messages = snapshot.values.get("messages", [])
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        unfinished = [ToolMessage(tool_call_id=c["id"], name=c["name"], status="error",
            content="本轮执行已停止，此调用未取得完整结果；涉及写入请先核查状态，不得直接重放。",
            artifact={"stop_reason": reason})
            for m in messages if isinstance(m, AIMessage)
            for c in [*m.tool_calls, *m.invalid_tool_calls] if c["id"] not in answered]
        final = AIMessage(content=text, additional_kwargs={"execution_stop": reason})
        await session.graph.aupdate_state(session.config, {"messages": [*unfinished, final]}, as_node="model")
        await session.graph.aupdate_state(session.config, None, as_node=END)
        return ExecutionResult(text, "partial" if evidence.successful_tools or evidence.delegated else "failed", reason)
    finally:
        budget = get_budget()
        if budget is not None:
            budget.release_delivery()
    messages = result.get("messages", []) if isinstance(result, dict) else []
    final = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
    text = _message_text(final)
    reason = result.get("execution_stop") if isinstance(result, dict) else None
    if not text.strip():
        return ExecutionResult("本轮未形成有效答复，请重试或缩小任务范围。", "failed", reason or "empty_delivery")
    plan_steps = result.get('task_plan', {}).get('steps', [])
    reported = set(result.get('research_delivered_tasks', []))
    reported.update(key for key, value in result.get('task_plan', {}).get('tasks', {}).items()
                    if value.get('reported') and not value.get('stale'))
    plan_incomplete = any(step['status'] != 'delivered' and step['id'] not in reported for step in plan_steps)
    research_partial = result.get('research_delivery_status') == 'partial'
    return ExecutionResult(text, "needs_input" if result.get("delivery_needs_input") else
                           "failed" if reason == "provider_refusal" else "partial" if reason or plan_incomplete or research_partial else "completed", reason,
                           bool(result.get("product_delivery_complete")))
