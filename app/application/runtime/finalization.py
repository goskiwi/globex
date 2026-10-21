"""执行后的持久化收尾；不重跑图或业务工具。"""
import logging


logger = logging.getLogger(__name__)


class RunPersistenceError(RuntimeError):
    """业务可能已经执行，但必要的会话或证据保存未完成。"""
    def __init__(self, message, *, session_saved=False):
        super().__init__(message)
        self.session_saved = session_saved


class RunFinalizer:
    def __init__(self, sessions, evidence_store):
        self._sessions, self._evidence_store = sessions, evidence_store

    async def finish(self, intent, text, presentations, persistence_guard, *, cancelled=False) -> bool:
        session_id = intent.shopping_session_id
        saved = False
        try:
            if persistence_guard is not None and not persistence_guard():
                await self._sessions.invalidate(session_id)
                return False
            persisted = await self._sessions.persist(session_id)
            if persisted is False:
                return False
            saved = True
            if self._evidence_store is not None:
                for accepted in presentations or ():
                    from app.application.usecases.intents import PRODUCT_DELIVERIES
                    if accepted['kind'] in PRODUCT_DELIVERIES:
                        from app.application.usecases.product_cards import display_groups
                        await self._evidence_store.save(intent.buyer_id, session_id, "display_batch", {
                            **accepted['payload'], 'presentation_kind': accepted['kind'],
                            'display_groups': display_groups(accepted['kind'], accepted['payload']['hits'])})
                await self._evidence_store.save(intent.buyer_id, session_id, "conversation", {
                    "buyer": intent.raw_query, "agent": text,
                })
            return True
        except Exception as error:
            if cancelled:
                # 保存故障不能覆盖取消信号，也不能把中断改成可自动重试的故障。
                logger.warning("中断收尾保存失败：%s", type(error).__name__)
                return saved
            raise RunPersistenceError("执行后的必要记录保存失败", session_saved=saved) from error
