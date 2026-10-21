"""图执行边界：模型流观察、预算停止、未完成调用收口与结果投影。"""
import asyncio

from opentelemetry import trace
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END
from langgraph.errors import GraphRecursionError

from app.application.runtime.events import ModelStarted, ModelFinished
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
            self.observer(ModelStarted(str(run_id)))

    async def on_llm_end(self, response, *, run_id, **kwargs):
        if self.observer:
            self.observer(ModelFinished(str(run_id)))
        self.interrupted = self.interrupted or any(
            (generation.generation_info or {}).get("finish_reason") == "interrupted"
            for group in response.generations for generation in group
        )

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if self.observer:
            self.observer(ModelFinished(str(run_id)))


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
    except (asyncio.CancelledError, ExecutionStopped, GraphRecursionError, ContextCapacityError) as error:
        cancelled = isinstance(error, asyncio.CancelledError)
        reason = 'cancelled' if cancelled else error.reason if isinstance(error, ExecutionStopped) else (
            "context_capacity" if isinstance(error, ContextCapacityError) else "step_limit")
        text = "本轮执行已中断；已完成操作保留，未完成调用不会自动重放。" if cancelled else evidence.stopped_text(reason)
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
        # 用户停止、服务中断也必须先关闭持久工具批次，再交给运行服务记录停止状态。
        if cancelled:
            raise
        from app.application.runtime.request_scope import ShoppingContext
        context=ShoppingContext.current()
        presentations=tuple(p for p in snapshot.values.get('presentations',[]) if not context or p['runId']==context.source_run_id)
        return ExecutionResult(text, "partial" if evidence.successful_tools or evidence.delegated or presentations else "failed", reason,presentations)
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
    from app.application.runtime.request_scope import ShoppingContext
    origin=ShoppingContext.current().source_run_id if ShoppingContext.current() else ''
    presentations=tuple(p for p in result.get('presentations',[]) if not origin or p['runId']==origin)
    from langchain_core.messages import HumanMessage
    context=ShoppingContext.current()
    latest=next((m for m in reversed(messages) if isinstance(m,HumanMessage) and context and m.name==context.buyer_id),None)
    request_id=(context.source_run_id or (latest.id if latest else '')) if context else ''
    # 交易任务的权威回执已经在原始工具消息中；不写入研究计划或伪造研究交付标记。
    operations=[m.artifact['data'] for m in messages if isinstance(m,ToolMessage)
                and m.name=='task_dispatch' and isinstance((m.artifact or {}).get('data'),dict)
                and m.artifact['data'].get('agent')=='trade_agent'
                and request_id and (m.artifact['data'].get('scope') or {}).get('request_id')==request_id]
    from app.application.runtime.research_plan import pending_plan_goals
    unfinished=(any(operation['status']!='completed' for operation in operations)
                or bool(pending_plan_goals(result.get('task_plan',{}),request_id)))

    return ExecutionResult(text, 'needs_input' if (any(p['status']=='needs_input' for p in presentations)
        or any(operation['status']=='needs_input' for operation in operations)) else
        'failed' if reason=='provider_refusal' else
        'partial' if reason or unfinished or any(p['status']=='partial' for p in presentations) else 'completed',reason,presentations)
