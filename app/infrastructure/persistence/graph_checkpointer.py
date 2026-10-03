"""复用既有会话 fencing，保护 LangGraph checkpoint 和中间写入。"""
import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.infrastructure.context import ShoppingContext
from app.domain.session.ports.session_store import StaleSessionWrite


class FencedSqliteSaver(AsyncSqliteSaver):
    session_store = None

    def __init__(self, conn, **kwargs):
        super().__init__(conn, **kwargs)
        self._write_scope_lock = asyncio.Lock()

    def _guard(self, config):
        context = ShoppingContext.current()
        thread_id = config["configurable"]["thread_id"]
        if (self.session_store is None or context is None
                or context.shopping_session_id != thread_id):
            raise StaleSessionWrite("checkpoint 写入缺少可信会话执行权")
        return self.session_store.guard_execution(
            thread_id, context.buyer_id, context.session_fence,
        )

    async def aput(self, config, checkpoint, metadata, new_versions):
        async with self._write_scope_lock, self._guard(config):
            try:
                return await super().aput(config, checkpoint, metadata, new_versions)
            except BaseException:
                await asyncio.shield(self.conn.rollback())
                raise

    async def aput_writes(self, config, writes, task_id, task_path=""):
        async with self._write_scope_lock, self._guard(config):
            try:
                await super().aput_writes(config, writes, task_id, task_path)
            except BaseException:
                await asyncio.shield(self.conn.rollback())
                raise

    async def adelete_thread(self, thread_id):
        async with self._write_scope_lock:
            await super().adelete_thread(thread_id)
