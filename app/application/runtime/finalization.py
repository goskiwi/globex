"""执行后的持久化收尾；不重跑图或业务工具。"""
import asyncio
import logging

from app.domain.session.ports.conversation_store import ConversationEventRecord, ConversationTurn

logger = logging.getLogger(__name__)


class RunPersistenceError(RuntimeError):
    """业务可能已经执行，但必要的会话或证据保存未完成。"""


class RunFinalizer:
    def __init__(self, sessions, bus, conversation_store, evidence_store):
        self._sessions, self._bus = sessions, bus
        self._conversation_store, self._evidence_store = conversation_store, evidence_store

    async def finish(self, intent, text, delivered, latency_ms, trace, persistence_guard, *, cancelled=False):
        session_id = intent.shopping_session_id
        try:
            if persistence_guard is not None and not persistence_guard():
                await self._sessions.invalidate(session_id)
                return
            persisted = await self._sessions.persist(session_id)
            if persisted is False:
                return
            record = self.record_conversation(intent, text, latency_ms, trace)
            if cancelled:
                task = asyncio.create_task(record)
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            else:
                await record
            if self._evidence_store is not None:
                if delivered is not None:
                    await self._evidence_store.save(intent.buyer_id, session_id, "display_batch", delivered)
                await self._evidence_store.save(intent.buyer_id, session_id, "conversation", {
                    "buyer": intent.raw_query, "agent": text,
                })
        except Exception as error:
            if cancelled:
                # 保存故障不能覆盖取消信号，也不能把中断改成可自动重试的故障。
                logger.warning("中断收尾保存失败：%s", type(error).__name__)
                return
            raise RunPersistenceError("执行后的必要记录保存失败") from error

    async def record_conversation(self, intent, final_text: str, latency_ms: int, trace) -> None:
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
