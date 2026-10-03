# -*- coding: utf-8 -*-
"""LangGraph 轮次编排：上下文装配、流式投影、审批恢复与持久执行。"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from opentelemetry import trace
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from langgraph.graph import END
from langgraph.errors import GraphRecursionError

from app.application.runtime.events import ReplyStart, approval_event
from app.domain.session.ports.conversation_store import ConversationEventRecord, ConversationStore, ConversationTurn
from app.infrastructure.budget import init_budget, get_budget
from app.infrastructure.capability_registry import CapabilityVersionChanged
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.operational_metrics import begin_request, finish_request
from app.infrastructure.prompt_registry import PromptContractChanged
from app.infrastructure.security.output_guard import audit_output
from app.application.runtime.errors import ContextCapacityError, ExecutionStopped
from app.application.runtime.handoff import TaskEvidence
from app.application.runtime.execution import ExecutionResult

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class SubmitIntentInput:
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str
    raw_query: str
    confirmations: tuple[dict, ...] = ()
    message_id: str = ""
    run_id: str = ""


@dataclass(frozen=True)
class SubmitIntentOutput:
    shopping_session_id: str
    final_text: str
    error: str | None = None
    error_code: str | None = None
    status: str = "completed"
    stop_reason: str | None = None
    product_delivery_complete: bool = False


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


class MainAgentOrchestrator:
    def __init__(
        self,
        sessions,
        bus: TradeEventBus,
        *,
        conversation_store: Optional[ConversationStore] = None,
        output_guard_enabled: bool = True,
        loop_detector: Any = None,
        token_budget_total: int = 0,
        session_lease_factory: Callable[..., Any] | None = None,
        trade_state_provider: Callable[[str, str], Any] | None = None,
        evidence_store: Any = None,
    ) -> None:
        self._sessions = sessions
        self._bus = bus
        self._conversation_store = conversation_store
        self._output_guard_enabled = output_guard_enabled
        self._loop_detector = loop_detector
        self._token_budget_total = token_budget_total
        self._session_lease_factory = session_lease_factory
        self._trade_state_provider = trade_state_provider
        self._evidence_store = evidence_store
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._native_observer: ContextVar[Callable[[Any], None] | None] = ContextVar(
            "globex_native_event_observer", default=None
        )

    async def available_skills(self, buyer_id: str | None = None) -> dict:
        factory = self._sessions._main_factory
        registry = factory.capability_registry
        if registry is None:
            raise RuntimeError("选购方案服务尚未配置")
        available = {tool.name for tool in [*factory._search_factory.build_tools(), *factory._trade_factory.build_tools()]}

        def read():
            digest = registry.version_fingerprint()
            metadata = registry.metadata(available_tools=available, expected_digest=digest)
            fields = ("id", "version", "title", "description", "scope", "content_hash", "expires_at")
            return {"capability_digest": digest, "skills": [{key: item[key] for key in fields} for item in metadata]}

        result = await asyncio.to_thread(read)
        if buyer_id and factory.buyer_skill_store is not None:
            result["skills"] = [
                *await asyncio.to_thread(factory.buyer_skill_store.list, buyer_id),
                *result["skills"],
            ]
        return result

    def _guard_final_text(self, session_id: str, text: str) -> str:
        if not self._output_guard_enabled or not text:
            return text
        safe, cleaned = audit_output(text)
        if not safe:
            self._bus.publish(session_id, "error", {"message": "输出审核命中内部信息，已脱敏后下发"})
        return cleaned

    async def handle_intent(
        self,
        intent: SubmitIntentInput,
        event_observer: Callable[[Any], None] | None = None,
        fresh_session: bool = False,
        persistence_guard: Callable[[], bool] | None = None,
    ) -> SubmitIntentOutput:
        lock = self._session_locks.setdefault(intent.shopping_session_id, asyncio.Lock())
        async with lock, AsyncExitStack() as stack:
            if self._session_lease_factory is not None:
                lease = await stack.enter_async_context(self._session_lease_factory(intent.shopping_session_id))
                caller_guard = persistence_guard
                persistence_guard = lambda: lease.is_valid() and (caller_guard is None or caller_guard())
                fresh_session = True
            if fresh_session:
                await self._sessions.invalidate(intent.shopping_session_id)
            token = self._native_observer.set(event_observer)
            metrics = begin_request()
            status = "failed"
            try:
                result = await self._handle_intent(intent, persistence_guard)
                status = result.status
                return result
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            finally:
                self._native_observer.reset(token)
                self._bus.publish(intent.shopping_session_id, "usage.summary", finish_request(metrics, status))

    @asynccontextmanager
    async def session_operation(self, session_id):
        """会话管理与执行共享同一进程锁及原Redis租约。"""
        lock=self._session_locks.setdefault(session_id,asyncio.Lock())
        if lock.locked():
            from app.infrastructure.ag_ui_journal import JournalConflict
            raise JournalConflict("对话正在执行或整理，请先停止或等待完成")
        async with lock,AsyncExitStack() as stack:
            if self._session_lease_factory is not None:
                await stack.enter_async_context(self._session_lease_factory(session_id,wait_timeout=0))
            yield

    async def _handle_intent(self, intent, persistence_guard) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
        context_token = ShoppingContext.set(ShoppingContextSnapshot(
            shopping_session_id=session_id,
            buyer_id=intent.buyer_id,
            locale=intent.locale,
            currency=intent.currency,
            source_message_id=intent.message_id,
            source_run_id=intent.run_id,
        ))
        init_budget(self._token_budget_total)
        started_at = time.monotonic()
        trace = self._bus.subscribe(session_id) if self._conversation_store else None
        final_text = ""
        delivered = None
        recommendation = None

        def collect_delivery(event):
            nonlocal recommendation
            if event.shopping_session_id == session_id and event.type in {"recommendation.result", "comparison.result"}:
                recommendation = event.payload

        try:
            session = await self._sessions.get_or_create(session_id)
            snapshot = await session.graph.aget_state(session.config)
            pending = snapshot.interrupts
            if intent.confirmations:
                if not pending:
                    raise ValueError("当前没有待确认的长期记忆操作")
                pending_by_id = {}
                for item in pending:
                    projected = approval_event([item])
                    if projected is None or len(projected.tool_calls) != 1:
                        raise ValueError("旧版批量审批需要迁移，未执行任何操作")
                    pending_by_id[projected.reply_id + ":" + projected.tool_calls[0].id] = item
                received_ids = [entry.get("interrupt_id") for entry in intent.confirmations]
                if (len(set(received_ids)) != len(received_ids)
                        or set(received_ids) != set(pending_by_id)
                        or any(type(entry.get("approved")) is not bool for entry in intent.confirmations)):
                    raise ValueError("确认与当前待处理操作不一致，请刷新会话")
                decisions_by_id = {
                    str(pending_by_id[entry["interrupt_id"]].id): {
                        "decisions": [{"type": "approve" if entry["approved"] else "reject"}]
                    }
                    for entry in intent.confirmations
                }
                pending_ids = {str(item.id) for item in pending}
                if set(decisions_by_id) != pending_ids:
                    raise ValueError("确认与当前待处理操作不一致，请刷新会话")
                resume = next(iter(decisions_by_id.values())) if len(pending) == 1 else decisions_by_id
                with observe_run_events(collect_delivery):
                    execution = await self._invoke_graph(session_id, session, Command(resume=resume))
                final_text = execution.text
                remaining = await session.graph.aget_state(session.config)
                if remaining.interrupts:
                    event = approval_event(remaining.interrupts)
                    observer = self._native_observer.get()
                    if event is not None and observer is not None:
                        observer(event)
                    final_text = "还有长期记忆变更待确认，请继续核对下方操作。"
                    execution = ExecutionResult(final_text, "needs_input")
                final_text = self._guard_final_text(session_id, final_text)
                delivered = recommendation if execution.product_delivery_complete else None
                self._bus.publish(session_id, "final.result", {"text": final_text, "status": execution.status, "stop_reason": execution.stop_reason})
                return SubmitIntentOutput(session_id, final_text, status=execution.status, stop_reason=execution.stop_reason,
                                          product_delivery_complete=execution.product_delivery_complete)
            if pending:
                event = approval_event(pending)
                observer = self._native_observer.get()
                if event is not None and observer is not None:
                    observer(event)
                return SubmitIntentOutput(session_id, "请先确认或拒绝待处理的长期记忆操作，再继续对话。", status="needs_input")

            messages = list(snapshot.values.get("messages", [])) if isinstance(snapshot.values, dict) else []
            trade_state = await self._trade_state_provider(intent.buyer_id, session_id) if self._trade_state_provider else {}
            has_trade_state = bool(trade_state.get("orders") or trade_state.get("pending_confirmations"))
            inputs = [HumanMessage(content=intent.raw_query, name=intent.buyer_id,
                                   id=intent.message_id or None)]
            if has_trade_state:
                inputs.insert(0, SystemMessage(content=(
                    "以下是服务端交易账本当前事实。待确认不等于已执行，不能代替买家批准。\n"
                    + json.dumps(trade_state, ensure_ascii=False)
                ), name="trade_state"))
            if self._evidence_store is not None:
                candidates = await self._evidence_store.search(intent.buyer_id, session_id, kind="display_batch", limit=1)
                if candidates:
                    latest = candidates[0]
                    inputs.insert(0, SystemMessage(content=(
                        "最近已交付商品的顺序仅用于解析指代；当前价格和库存仍须重新核验。\n"
                        + json.dumps({"result_ref": latest["result_ref"], "candidates": [
                            {"position": index + 1, "product_id": item["product_id"], "title": item["title"],
                             "sku_id": item.get("default_sku_id"), "quantity": item.get("quantity", 1)}
                            for index, item in enumerate(latest["data"].get("hits", []))
                        ]}, ensure_ascii=False)
                    ), name="candidate_state"))

            await self._remove_stale_runtime_hints(session, messages)
            with observe_run_events(collect_delivery):
                execution = await self._reply(session_id, session, inputs)
                final_text = execution.text
            final_text = self._guard_final_text(session_id, final_text)
            state_after = await session.graph.aget_state(session.config)
            if state_after.interrupts:
                event = approval_event(state_after.interrupts)
                observer = self._native_observer.get()
                if event is not None and observer is not None:
                    observer(event)
                final_text = "这次长期记忆变更还未执行，请在下方确认或拒绝。"
                execution = ExecutionResult(final_text, "needs_input")
            delivered = recommendation if execution.product_delivery_complete else None
            self._bus.publish(session_id, "final.result", {"text": final_text, "status": execution.status, "stop_reason": execution.stop_reason})
            return SubmitIntentOutput(session_id, final_text, status=execution.status, stop_reason=execution.stop_reason,
                                      product_delivery_complete=execution.product_delivery_complete)
        except ContextCapacityError:
            text = "本次比较的内容超过安全上下文容量。原始记录已保留，请缩小商品范围或分批比较。"
            final_text = text
            return SubmitIntentOutput(session_id, text, text, "CONTEXT_CAPACITY_EXCEEDED", status="failed", stop_reason="context_capacity")
        except (CapabilityVersionChanged, PromptContractChanged):
            text = "选购环境已更新，旧记录仍保留。请在新会话中继续本次需求。"
            return SubmitIntentOutput(session_id, text, text, "SESSION_VERSION_CHANGED", status="failed", stop_reason="session_version_changed")
        except asyncio.CancelledError:
            final_text = "[cancelled] 本轮执行已中断"
            self._bus.publish(session_id, "error", {"message": "本轮执行已中断", "cancelled": True})
            raise
        except Exception as error:
            logger.exception("LangGraph 主图异常")
            message = "本轮服务暂时无法完成，请稍后重试；涉及写入请先核查状态。"
            self._bus.publish(session_id, "error", {"message": message})
            return SubmitIntentOutput(session_id, f"[error] {message}", message, status="failed", stop_reason="execution_error")
        finally:
            try:
                if persistence_guard is None or persistence_guard():
                    persisted = await self._sessions.persist(session_id)
                    if persisted is not False:
                        record = self._record_conversation(
                            intent, final_text, int((time.monotonic() - started_at) * 1000), trace
                        )
                        if final_text.startswith("[cancelled]"):
                            record_task = asyncio.create_task(record)
                            try:
                                await asyncio.shield(record_task)
                            except asyncio.CancelledError:
                                await record_task
                                raise
                        else:
                            await record
                        if self._evidence_store is not None:
                            if delivered is not None:
                                await self._evidence_store.save(intent.buyer_id, session_id, "display_batch", delivered)
                            await self._evidence_store.save(intent.buyer_id, session_id, "conversation", {
                                "buyer": intent.raw_query,
                                "agent": final_text,
                            })
                else:
                    await self._sessions.invalidate(session_id)
            finally:
                if trace is not None:
                    self._bus.unsubscribe(session_id, trace)
                ShoppingContext.reset(context_token)

    async def _remove_stale_runtime_hints(self, session, messages: list[BaseMessage]) -> None:
        stale = [message for message in messages if getattr(message, "name", None) in {
            "memory_hint", "trade_state", "candidate_state", "selected_skill_reference"
        }]
        if stale:
            await session.graph.aupdate_state(
                session.config,
                {"messages": [RemoveMessage(id=message.id) for message in stale]},
            )

    async def _invoke_graph(self, session_id: str, session, graph_input) -> ExecutionResult:
        evidence = TaskEvidence(session_id)
        observer = _GraphStreamObserver(self._native_observer.get())
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
        return ExecutionResult(text, "needs_input" if result.get("delivery_needs_input") else
                               "failed" if reason == "provider_refusal" else "partial" if reason else "completed", reason,
                               bool(result.get("product_delivery_complete")))

    async def _reply(self, session_id: str, session, inputs: list[BaseMessage]) -> ExecutionResult:
        # 模型故障只在模型中间件重试；不在外层自动重放整轮。
        return await self._invoke_graph(session_id, session, {"messages": inputs})

    async def _record_conversation(self, intent, final_text: str, latency_ms: int, trace) -> None:
        if self._conversation_store is None:
            return
        events: list[ConversationEventRecord] = []
        if trace is not None:
            self._bus.unsubscribe(intent.shopping_session_id, trace)
            while not trace.empty():
                event = trace.get_nowait()
                events.append(ConversationEventRecord(
                    session_id=intent.shopping_session_id,
                    type=event.type,
                    payload=event.payload if isinstance(event.payload, dict) else {"value": event.payload},
                    occurred_at=event.occurred_at,
                ))
        try:
            await self._conversation_store.touch_session(
                intent.shopping_session_id, intent.buyer_id, intent.locale, intent.currency
            )
            await self._conversation_store.append_turn(ConversationTurn(
                session_id=intent.shopping_session_id,
                buyer_id=intent.buyer_id,
                role="buyer",
                content=intent.raw_query,
            ))
            await self._conversation_store.append_turn(ConversationTurn(
                session_id=intent.shopping_session_id,
                buyer_id=intent.buyer_id,
                role="agent",
                content=final_text,
                latency_ms=latency_ms,
            ))
            await self._conversation_store.append_events(events)
        except Exception as error:
            logger.warning("对话记录写入失败：%s", error)
