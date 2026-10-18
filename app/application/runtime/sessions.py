"""同进程会话操作协调；等待者也持有引用，避免回收后产生第二把锁。"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


class SessionBusy(ValueError):
    pass


@dataclass
class _Entry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class SessionPermit:
    def __init__(self, coordinator, session_id, entry):
        self._coordinator, self._session_id, self._entry = coordinator, session_id, entry
        self._released = False

    def release(self):
        if not self._released:
            self._released = True
            self._entry.lock.release()
            self._coordinator._drop(self._session_id, self._entry)


class SessionCoordinator:
    def __init__(self):
        self._entries: dict[str, _Entry] = {}

    @property
    def active_sessions(self) -> int:
        return len(self._entries)

    def _drop(self, session_id, entry):
        entry.users -= 1
        if entry.users == 0:
            self._entries.pop(session_id)

    async def acquire(self, session_id: str, *, wait=True) -> SessionPermit:
        entry = self._entries.setdefault(session_id, _Entry())
        if not wait and entry.users:
            raise SessionBusy("对话正在执行或整理，请先停止或等待完成")
        entry.users += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            self._drop(session_id, entry)
            raise
        return SessionPermit(self, session_id, entry)

    @asynccontextmanager
    async def hold(self, session_id: str, *, wait=True):
        permit = await self.acquire(session_id, wait=wait)
        try:
            yield permit
        finally:
            permit.release()
