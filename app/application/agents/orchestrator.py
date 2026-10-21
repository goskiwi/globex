# -*- coding: utf-8 -*-
"""LangGraph 轮次编排：上下文装配、流式投影、审批恢复与持久执行。"""
from __future__ import annotations

import asyncio
import json
import uuid
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.types import Command

from app.application.runtime.events import approval_event
from app.domain.session.ports.conversation_store import ConversationStore
from app.infrastructure.budget import init_budget
from app.infrastructure.capability_registry import CapabilityVersionChanged
from app.application.runtime.request_scope import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.operational_metrics import begin_request, finish_request
from app.infrastructure.prompt_registry import PromptContractChanged
from app.application.runtime.errors import ContextCapacityError
from app.application.runtime.execution import ExecutionResult
from app.application.usecases.intents import SubmitIntentInput, SubmitIntentOutput
from app.application.runtime.sessions import SessionCoordinator, SessionBusy
from app.application.runtime.finalization import RunFinalizer, RunPersistenceError
from app.application.runtime.graph_execution import execute_graph

logger = logging.getLogger(__name__)


class MainAgentOrchestrator:
    def __init__(
        self,
        sessions,
        bus: TradeEventBus,
        *,
        conversation_store: Optional[ConversationStore] = None,
        loop_detector: Any = None,
        token_budget_total: int = 0,
        session_lease_factory: Callable[..., Any] | None = None,
        trade_state_provider: Callable[[str, str], Any] | None = None,
        evidence_store: Any = None,
        buyer_workspace=None,
    ) -> None:
        self._sessions = sessions
        self.buyer_workspace = buyer_workspace
        self._bus = bus
        self._conversation_store = conversation_store
        self._loop_detector = loop_detector
        self._token_budget_total = token_budget_total
        self._session_lease_factory = session_lease_factory
        self._trade_state_provider = trade_state_provider
        self._evidence_store = evidence_store
        self.session_coordinator = SessionCoordinator()
        self._native_observer: ContextVar[Callable[[Any], None] | None] = ContextVar(
            "globex_native_event_observer", default=None
        )

    async def available_skills(self, buyer_id: str | None = None) -> dict:
        if self.buyer_workspace is None:
            raise RuntimeError("选购方案服务尚未配置")
        return await self.buyer_workspace.available_skills(buyer_id)

    @property
    def sessions(self):
        return self._sessions

    @asynccontextmanager
    async def session_lease(self, session_id, *, wait=True):
        """所有会话操作复用既有跨进程租约；单进程模式返回空租约。"""
        async with AsyncExitStack() as stack:
            lease = None
            if self._session_lease_factory is not None:
                options = {} if wait else {"wait_timeout": 0}
                lease = await stack.enter_async_context(self._session_lease_factory(session_id, **options))
            yield lease

    def _guard_final_text(self, session_id: str, text: str) -> str:
        cleaned = self._bus.projection.business(text)
        if cleaned != text:
            self._bus.publish(session_id, "error", {"message": "输出审核命中内部信息，已脱敏后下发"})
        return cleaned

    async def handle_intent(
        self,
        intent: SubmitIntentInput,
        event_observer: Callable[[Any], None] | None = None,
        fresh_session: bool = False,
        persistence_guard: Callable[[], bool] | None = None,
    ) -> SubmitIntentOutput:
        async with self.session_coordinator.hold(intent.shopping_session_id), self.session_lease(intent.shopping_session_id) as lease:
            if lease is not None:
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
        try:
            async with self.session_coordinator.hold(session_id, wait=False), self.session_lease(session_id, wait=False):
                yield
        except SessionBusy as error:
            from app.infrastructure.ag_ui_journal import JournalConflict
            raise JournalConflict(str(error)) from error

    async def _handle_intent(self, intent, persistence_guard) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
        context_token = ShoppingContext.set(ShoppingContextSnapshot(
            shopping_session_id=session_id, buyer_id=intent.buyer_id,
            locale=intent.locale, currency=intent.currency,
            source_message_id=intent.message_id, source_run_id=intent.run_id or uuid.uuid4().hex,
        ))
        init_budget(self._token_budget_total)
        started_at = time.monotonic()
        trace = self._bus.subscribe(session_id) if self._conversation_store else None
        finalizer = RunFinalizer(self._sessions, self._bus, self._conversation_store, self._evidence_store)
        accepted=[]
        def collect(event):
            if event.shopping_session_id==session_id and event.type=='presentation.result':
                accepted.append(event.payload)
        try:
            try:
                with observe_run_events(collect):
                    result = await self._execute_intent(intent)
            except asyncio.CancelledError:
                text = "[cancelled] 本轮执行已中断"
                saved = await finalizer.finish(intent, text, accepted, persistence_guard, cancelled=True)
                if saved:
                    await finalizer.record_conversation(intent, text,
                        int((time.monotonic() - started_at) * 1000), trace, cancelled=True)
                raise
            try:
                saved = await finalizer.finish(intent, result.final_text, accepted, persistence_guard)
            except RunPersistenceError as error:
                logger.exception("执行收尾失败，不自动重放业务操作")
                saved = error.session_saved
                message = "本轮操作可能已生效，但执行记录保存未完成；请先核查订单、确认单和会话状态，不要直接重复操作。"
                result = SubmitIntentOutput(session_id, message, message,
                    "PERSISTENCE_FAILED", status="failed", stop_reason="persistence_error")
            # 必要保存完成后发布终态，但执行身份和审计订阅仍然有效。
            self._bus.publish(session_id, "final.result", {
                "text": result.final_text, "status": result.status, "stop_reason": result.stop_reason})
            if saved:
                await finalizer.record_conversation(intent, result.final_text,
                    int((time.monotonic() - started_at) * 1000), trace)
            return result
        finally:
            if trace is not None:
                self._bus.unsubscribe(session_id, trace)
            ShoppingContext.reset(context_token)

    async def _execute_intent(self, intent) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
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
                return SubmitIntentOutput(session_id, final_text, status=execution.status, stop_reason=execution.stop_reason,
                                          presentations=execution.presentations)
            if pending:
                event = approval_event(pending)
                observer = self._native_observer.get()
                if event is not None and observer is not None:
                    observer(event)
                final_text = "请先确认或拒绝待处理的长期记忆操作，再继续对话。"
                return SubmitIntentOutput(session_id, final_text, status="needs_input")

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
                        + json.dumps({"result_ref": latest["result_ref"],
                            "candidates": latest["data"]["display_groups"]}, ensure_ascii=False)
                    ), name="candidate_state"))

            await self._remove_stale_runtime_hints(session, messages)
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
            return SubmitIntentOutput(session_id, final_text, status=execution.status, stop_reason=execution.stop_reason,
                                      presentations=execution.presentations)
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
        return await execute_graph(session_id, session, graph_input, self._native_observer.get())

    async def _reply(self, session_id: str, session, inputs: list[BaseMessage]) -> ExecutionResult:
        # 模型故障只在模型中间件重试；不在外层自动重放整轮。
        return await self._invoke_graph(session_id, session, {"messages": inputs})
