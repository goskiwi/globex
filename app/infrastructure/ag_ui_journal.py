# -*- coding: utf-8 -*-
"""AG-UI 持久事件日志：SQLite 事务内追加序号、更新投影并绑定买家归属。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import time
import uuid
from typing import Any

import aiosqlite
from app.application.runtime.outcomes import apply_run_error, error_events


class JournalConflict(ValueError):
    pass


class JournalForbidden(PermissionError):
    pass


class JournalNotFound(LookupError):
    pass


class JournalLeaseLost(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class AGUIJournal:
    def __init__(self, path: str | Path, *, projection=None):
        from app.infrastructure.security.public_projection import PublicProjection
        self.projection = projection or PublicProjection()
        self.path = Path(path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        async with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 初始化短等待；总体期限避免 busy_timeout × 重试次数形成长时间挂起。
            deadline = time.monotonic() + 8.0
            for attempt in range(20):
                try:
                    # 每次失败都关闭连接，不能带着上一轮未完成游标/事务重试。
                    async with aiosqlite.connect(self.path, timeout=0.5) as db:
                        async with db.execute("PRAGMA busy_timeout=500") as cursor:
                            await cursor.fetchall()
                        # PRAGMA 会返回结果行；显式耗尽并关闭，避免游标持锁阻碍后续 DDL。
                        async with db.execute("PRAGMA journal_mode=WAL") as cursor:
                            await cursor.fetchall()
                        await db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS agui_sessions (
                  session_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL, title TEXT NOT NULL,
                  messages_json TEXT NOT NULL, state_json TEXT NOT NULL, last_run_id TEXT NOT NULL,
                  updated_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS agui_buyer_sessions ON agui_sessions(buyer_id,updated_at);
                CREATE TABLE IF NOT EXISTS agui_runs (
                  run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, buyer_id TEXT NOT NULL,
                  fingerprint TEXT NOT NULL, input_json TEXT NOT NULL, projection_json TEXT NOT NULL,
                  status TEXT NOT NULL, owner TEXT NOT NULL, lease_until REAL NOT NULL,
                  last_seq INTEGER NOT NULL DEFAULT 0, stop_requested INTEGER NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS agui_run_session ON agui_runs(session_id,status);
                CREATE TABLE IF NOT EXISTS agui_events (
                  run_id TEXT NOT NULL, seq INTEGER NOT NULL, event_json TEXT NOT NULL,
                  PRIMARY KEY(run_id,seq));
                COMMIT;
                        """)
                    async with aiosqlite.connect(self.path, timeout=0.5) as db:
                        await db.execute("BEGIN IMMEDIATE")
                        async with db.execute("PRAGMA table_info(agui_runs)") as cursor:
                            columns = {row[1] for row in await cursor.fetchall()}
                        for name in ("request_id", "request_scope"):
                            if name not in columns:
                                await db.execute(f"ALTER TABLE agui_runs ADD COLUMN {name} TEXT")
                        await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS agui_request_identity ON agui_runs(buyer_id,request_scope,request_id) WHERE request_id IS NOT NULL")
                        await db.commit()
                    self._initialized = True
                    return
                except aiosqlite.OperationalError as err:
                    if (not any(marker in str(err).lower() for marker in ("locked", "busy"))
                            or attempt == 19 or time.monotonic() >= deadline):
                        raise
                    await asyncio.sleep(min(0.05, max(0, deadline - time.monotonic())))

    @asynccontextmanager
    async def _db(self, write=False):
        await self.initialize()
        async with aiosqlite.connect(self.path, timeout=10) as db:
            db.row_factory = aiosqlite.Row
            if write:
                await db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                if write:
                    await db.commit()
            except BaseException:
                if write:
                    await db.rollback()
                raise

    @asynccontextmanager
    async def _read_snapshot(self):
        """恢复只占用短写事务；完整历史组装使用一致的只读快照。"""
        await self.recover_expired()
        async with self._db() as db:
            await db.execute("BEGIN")
            yield db

    @staticmethod
    async def _one(db, sql, params=()):
        async with db.execute(sql, params) as cursor:
            return await cursor.fetchone()

    @staticmethod
    def _owned(row, buyer_id, session_id=None):
        if row is None:
            raise JournalNotFound("运行或会话不存在")
        if row["buyer_id"] != buyer_id or (session_id is not None and row["session_id"] != session_id):
            raise JournalForbidden("无权访问该买家的会话或运行")
        return row

    @staticmethod
    def _run(row):
        projection = json.loads(row["projection_json"])
        pending = row['status'] in {'queued','running'}
        state = projection['state']
        if state.get('executionError'):
            apply_run_error(state, state['executionError'], state.get('errorCode'))
        final = next((m.get('content','') for m in reversed(projection['messages'])
                      if m.get('id') == row['run_id']+':final:answer'), '')
        result = None if pending else {'text':state.get('executionError') or final,
            'status':state.get('executionStatus') or ('completed' if row['status']=='completed' else 'failed'),
            'stop_reason':state.get('stopReason') or state.get('errorCode'), 'presentations':state.get('presentations',[])}
        return {"result":result, "runId": row["run_id"], "threadId": row["session_id"], "status": row["status"],
                "cursor": row["last_seq"], "stopRequested": bool(row["stop_requested"]),
                "input": json.loads(row["input_json"]), "messages": projection["messages"],
                "state": projection["state"], "updatedAt": int(row["updated_at"] * 1000)}

    async def reserve(self, body: dict, buyer_id: str, owner: str, lease_seconds=30, *, queued=False, request_id=None) -> tuple[dict, bool]:
        now, run_id, session_id = time.time(), body["runId"], body["threadId"]
        request_scope = session_id or ""
        user = body["messages"][-1]
        props = body.get("forwardedProps") or {}
        identity = {"threadId": session_id, "buyerId": buyer_id, "message": user,
                    "locale": props.get("locale", "zh-CN"), "currency": props.get("currency", "CNY")}
        if request_id is not None:
            identity["message"] = {"role":"user", "content":user["content"]}
        if body.get("resume"):
            identity["resume"] = body["resume"]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        async with self._db(True) as db:
            await self._recover(db, now)
            previous = (await self._one(db, "SELECT * FROM agui_runs WHERE buyer_id=? AND request_scope=? AND request_id=?",
                                       (buyer_id, request_scope, request_id)) if request_id is not None else
                        await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,)))
            if previous:
                self._owned(previous, buyer_id, session_id)
                if previous["fingerprint"] != fingerprint:
                    raise JournalConflict("相同 runId 已用于不同请求")
                return self._run(previous), False
            if session_id is None:
                session_id = "session-" + uuid.uuid4().hex
                body = {**body, "threadId":session_id}
            session = await self._one(db, "SELECT * FROM agui_sessions WHERE session_id=?", (session_id,))
            if session:
                self._owned(session, buyer_id)
            running = await self._one(db, "SELECT run_id FROM agui_runs WHERE session_id=? AND status IN ('queued','running')", (session_id,))
            if running:
                raise JournalConflict("该会话仍有执行中的运行，请先恢复或明确停止它")
            previous_state=json.loads(session["state_json"]) if session else {}
            if body.get("resume"):
                pending={p['id'] for p in previous_state.get('toolApprovals',[])}
                requested=[p['interruptId'] for p in body['resume']]
                if not requested or len(set(requested))!=len(requested) or not set(requested)<=pending:
                    raise JournalConflict("确认已处理或与当前待执行操作不一致，请刷新会话")
            trusted_state = {}
            if body.get('resume') and 'skillUsages' in previous_state:
                trusted_state['skillUsages'] = previous_state['skillUsages']
            # 仅新接纳分配身份；重试和 worker 复用已持久化的输入。
            user = {**user, "id": uuid.uuid4().hex}
            messages = json.loads(session["messages_json"]) if session else []
            # 客户端历史/state 不是事实来源；只接受本轮 user，旧历史由日志恢复。
            messages = [*messages[-99:], {"id": user["id"], "role": "user", "content": user["content"]}]
            body = {**body, "messages": messages, "state": trusted_state}
            projection = {"messages": messages, "state": trusted_state, "openMessages": [], "openTools": []}
            await db.execute("INSERT INTO agui_runs(run_id,session_id,buyer_id,fingerprint,input_json,projection_json,status,owner,lease_until,created_at,updated_at,request_id,request_scope) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, session_id, buyer_id, fingerprint, _json(body), _json(projection), "queued" if queued else "running",
                 owner, now + lease_seconds, now, now, request_id, request_scope if request_id is not None else None))
            title = session["title"] if session else str(user["content"])[:48]
            await db.execute("INSERT INTO agui_sessions VALUES(?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET messages_json=excluded.messages_json,state_json=excluded.state_json,last_run_id=excluded.last_run_id,updated_at=excluded.updated_at",
                (session_id, buyer_id, title, _json(messages), _json(trusted_state), run_id, now))
            if queued:
                row = await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,))
                initial = {**trusted_state, "status":"queued", "process": {
                    "runId":run_id, "userMessageId":user["id"], "status":"queued", "steps":[]}}
                await self._append(db, row, [{"type":"RUN_STARTED", "threadId":session_id, "runId":run_id},
                                            {"type":"MESSAGES_SNAPSHOT", "messages":messages},
                                            {"type":"STATE_SNAPSHOT", "snapshot":initial}])
            return self._run(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,))), True

    @staticmethod
    def _project(projection, event):
        kind = event["type"]
        if kind == "MESSAGES_SNAPSHOT":
            projection["messages"] = [m for m in event["messages"] if m.get("role") in {"user", "assistant"}]
        elif kind == "STATE_SNAPSHOT":
            projection["state"] = event["snapshot"]
        elif kind == "TEXT_MESSAGE_START":
            message_id = event["messageId"]
            if not any(m["id"] == message_id for m in projection["messages"]):
                projection["messages"].append({"id": message_id, "role": "assistant", "content": ""})
            if message_id not in projection["openMessages"]:
                projection["openMessages"].append(message_id)
        elif kind == "TEXT_MESSAGE_CONTENT":
            message = next((m for m in projection["messages"] if m["id"] == event["messageId"]), None)
            if message is None:
                raise JournalConflict("消息增量必须在对应 START 之后")
            message["content"] += event["delta"]
        elif kind == "TEXT_MESSAGE_END":
            projection["openMessages"] = [i for i in projection["openMessages"] if i != event["messageId"]]
        elif kind == "RUN_ERROR":
            apply_run_error(projection['state'], event['message'], event.get('code'))
        elif kind == "TOOL_CALL_START":
            projection["openTools"].append(event["toolCallId"])
        elif kind == "TOOL_CALL_END":
            projection["openTools"] = [i for i in projection["openTools"] if i != event["toolCallId"]]

    async def _append(self, db, row, events):
        projection, seq, status = json.loads(row["projection_json"]), row["last_seq"], row["status"]
        for event in events:
            self._project(projection, event)
            seq += 1
            await db.execute("INSERT INTO agui_events VALUES(?,?,?)", (row["run_id"], seq, _json(event)))
            if event["type"] == "RUN_FINISHED":
                status = "completed"
            elif event["type"] == "RUN_ERROR":
                status = "stopped" if event.get("code") == "CANCELLED" else "interrupted" if event.get("code") == "SERVER_RESTART" else "error"
        now = time.time()
        await db.execute("UPDATE agui_runs SET projection_json=?,last_seq=?,status=?,updated_at=? WHERE run_id=?",
                         (_json(projection), seq, status, now, row["run_id"]))
        await db.execute("UPDATE agui_sessions SET messages_json=?,state_json=?,updated_at=? WHERE session_id=? AND last_run_id=?",
                         (_json(projection["messages"][-100:]), _json(projection["state"]), now, row["session_id"], row["run_id"]))

    async def append(self, run_id, owner, events):
        async with self._db(True) as db:
            row = await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,))
            if not row or row["owner"] != owner or row["status"] != "running" or row["lease_until"] <= time.time():
                raise JournalLeaseLost("运行日志执行租约已失效")
            await self._append(db, row, events)

    async def renew(self, run_id, owner, lease_seconds):
        async with self._db(True) as db:
            cursor = await db.execute("UPDATE agui_runs SET lease_until=? WHERE run_id=? AND owner=? AND status IN ('queued','running') AND stop_requested=0 AND lease_until>?",
                (time.time() + lease_seconds, run_id, owner, time.time()))
            return cursor.rowcount == 1

    async def _recover(self, db, now):
        async with db.execute("SELECT * FROM agui_runs WHERE status='running' AND lease_until<=?", (now,)) as cursor:
            expired = await cursor.fetchall()
        for row in expired:
            projection = json.loads(row["projection_json"])
            stopped = bool(row["stop_requested"])
            events = [{"type": "TEXT_MESSAGE_END", "messageId": i} for i in projection["openMessages"]]
            if row["last_seq"] == 0:
                events.insert(0, {"type": "RUN_STARTED", "threadId": row["session_id"], "runId": row["run_id"]})
            events += [{"type": "TOOL_CALL_END", "toolCallId": i} for i in projection["openTools"]]
            events += error_events(projection['state'], "本轮已明确停止" if stopped else
                "服务重启或执行租约失效，本轮未完成；已保存内容仍可恢复，涉及写入请先核查状态，不直接重放。",
                "CANCELLED" if stopped else "SERVER_RESTART")
            await self._append(db, row, events)

    async def recover_expired(self):
        async with self._db(True) as db:
            await self._recover(db, time.time())

    async def run(self, run_id, buyer_id, session_id=None):
        async with self._read_snapshot() as db:
            return self._run(self._owned(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,)), buyer_id, session_id))

    async def claim(self, run_id, buyer_id, owner, lease_seconds):
        async with self._db(True) as db:
            await self._recover(db, time.time())
            row = self._owned(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,)), buyer_id)
            if row['status'] != 'queued':
                return self._run(row), False
            await db.execute("UPDATE agui_runs SET status='running',owner=?,lease_until=? WHERE run_id=?",
                             (owner, time.time()+lease_seconds, run_id))
            return self._run(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,))), True

    async def queued(self):
        async with self._db() as db:
            async with db.execute("SELECT * FROM agui_runs WHERE status='queued' ORDER BY created_at,run_id") as cursor:
                return [self._run(row) for row in await cursor.fetchall()]

    async def events(self, run_id, buyer_id, after=0, limit=200):
        async with self._read_snapshot() as db:
            row = self._owned(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,)), buyer_id)
            if after < 0 or after > row["last_seq"]:
                raise JournalConflict("事件游标超出该运行范围")
            async with db.execute("SELECT seq,event_json FROM agui_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, after, limit)) as cursor:
                events = [{"seq": item["seq"], "event": self.projection.event(json.loads(item["event_json"]))} for item in await cursor.fetchall()]
            return events, row["status"], row["last_seq"]

    async def request_stop(self, run_id, buyer_id):
        async with self._db(True) as db:
            row = self._owned(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,)), buyer_id)
            if row["status"] in {"queued", "running"}:
                await db.execute("UPDATE agui_runs SET stop_requested=1 WHERE run_id=?", (run_id,))
                if row['status'] == 'queued':
                    projection=json.loads(row['projection_json']); state=projection['state']
                    await self._append(db,row,error_events(state, '本轮已在排队阶段停止，未执行模型或业务工具。', 'CANCELLED'))
        return await self.run(run_id, buyer_id)

    async def end_owned(self, run_id, owner, *, stopped=False):
        """生产协程在首次调度前就被取消时也要收口；终态不覆盖，其他 owner 不可写。"""
        async with self._db(True) as db:
            row = await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (run_id,))
            if not row or row["owner"] != owner or row["status"] != "running":
                return
            projection = json.loads(row["projection_json"])
            events = [{"type": "TEXT_MESSAGE_END", "messageId": i} for i in projection["openMessages"]]
            if row["last_seq"] == 0:
                events.insert(0, {"type": "RUN_STARTED", "threadId": row["session_id"], "runId": row["run_id"]})
            events += [{"type": "TOOL_CALL_END", "toolCallId": i} for i in projection["openTools"]]
            events += error_events(projection['state'], "本轮已明确停止" if stopped else
                "服务已关闭，本轮已保存为中断；涉及写入请先核查状态，不直接重放。",
                "CANCELLED" if stopped else "SERVER_RESTART")
            await self._append(db, row, events)

    async def sessions(self, buyer_id):
        async with self._read_snapshot() as db:
            # 这里只返回轻量目录，不能静默隐藏第 101 个及更早的会话。
            async with db.execute("SELECT session_id,title,updated_at,last_run_id FROM agui_sessions WHERE buyer_id=? ORDER BY updated_at DESC", (buyer_id,)) as cursor:
                return [{"id": row["session_id"], "title": row["title"], "updatedAt": int(row["updated_at"] * 1000), "runId": row["last_run_id"], "source": "server"} for row in await cursor.fetchall()]

    async def rename_session(self, session_id, buyer_id, title):
        title=title.strip()
        if not title or len(title)>100:
            raise ValueError("标题需要1到100个字")
        async with self._db(True) as db:
            self._owned(await self._one(db,"SELECT * FROM agui_sessions WHERE session_id=?",(session_id,)),buyer_id)
            await db.execute("UPDATE agui_sessions SET title=? WHERE session_id=? AND buyer_id=?",(title,session_id,buyer_id))
        return {"id":session_id,"title":title}

    async def require_idle(self, session_id,buyer_id):
        async with self._db(True) as db:
            await self._recover(db,time.time())
            self._owned(await self._one(db,"SELECT * FROM agui_sessions WHERE session_id=?",(session_id,)),buyer_id)
            if await self._one(db,"SELECT run_id FROM agui_runs WHERE session_id=? AND status IN ('queued','running')",(session_id,)):
                raise JournalConflict("对话仍在运行，请先停止后再删除")

    async def delete_session(self, session_id, buyer_id, cleanup):
        # 持有原日志写事务，防止清理过程中同一对话又被reserve或append写回。
        async with self._db(True) as db:
            await self._recover(db,time.time())
            self._owned(await self._one(db,"SELECT * FROM agui_sessions WHERE session_id=?",(session_id,)),buyer_id)
            if await self._one(db,"SELECT run_id FROM agui_runs WHERE session_id=? AND status IN ('queued','running')",(session_id,)):
                raise JournalConflict("对话仍在运行，请先停止后再删除")
            await cleanup()
            await db.execute("DELETE FROM agui_events WHERE run_id IN (SELECT run_id FROM agui_runs WHERE session_id=? AND buyer_id=?)",(session_id,buyer_id))
            await db.execute("DELETE FROM agui_runs WHERE session_id=? AND buyer_id=?",(session_id,buyer_id))
            await db.execute("DELETE FROM agui_sessions WHERE session_id=? AND buyer_id=?",(session_id,buyer_id))
        return {"deleted":True,"id":session_id}

    async def session(self, session_id, buyer_id):
        async with self._read_snapshot() as db:
            row = self._owned(await self._one(db, "SELECT * FROM agui_sessions WHERE session_id=?", (session_id,)), buyer_id)
            run = self._run(await self._one(db, "SELECT * FROM agui_runs WHERE run_id=?", (row["last_run_id"],)))
            # sessions.messages_json 是下一轮使用的滑动窗口，不是页面完整历史。
            # 旧版本已保留每轮最终投影；按持久顺序合并即可恢复，无需重写原始数据。
            messages, presentations, processes = {}, {}, []
            async with db.execute(
                "SELECT run_id,status,projection_json,updated_at FROM agui_runs "
                "WHERE session_id=? AND buyer_id=? ORDER BY created_at,rowid",
                (session_id, buyer_id),
            ) as cursor:
                async for saved in cursor:
                    projection = json.loads(saved["projection_json"])
                    for message in projection["messages"]:
                        messages[message["id"]] = message
                    state = projection["state"]
                    if state.get('executionError'):
                        apply_run_error(state, state['executionError'], state.get('errorCode'))
                    process = state.get('process')
                    if isinstance(process,dict) and process.get('runId')==saved['run_id']:
                        # 进程中断后未必收到终态快照；只据日志的真实终态标为未确认，不伪装成功。
                        if saved['status'] in {'interrupted','error','stopped'} and process.get('status') in {'queued','running'}:
                            process['status']={'interrupted':'interrupted','error':'failed','stopped':'cancelled'}[saved['status']]
                            for step in process.get('steps',[]):
                                if step.get('status')=='running':
                                    step.update(status='unconfirmed',summary='运行已结束，未取得完整执行结果')
                        processes.append(process)
                    for accepted in state.get('presentations',[]):
                        presentations[accepted['id']]=accepted
            # 只回传最近 80 条事件的安全元数据，不暴露工具参数、结果或推理原文。
            async with db.execute(
                "SELECT r.run_id,e.seq,json_extract(e.event_json,'$.type') AS type,"
                "json_extract(e.event_json,'$.timestamp') AS timestamp "
                "FROM agui_events e JOIN agui_runs r ON r.run_id=e.run_id "
                "WHERE r.session_id=? AND r.buyer_id=? "
                "AND json_extract(e.event_json,'$.type') NOT IN ('TEXT_MESSAGE_CONTENT','TOOL_CALL_ARGS') "
                "ORDER BY r.created_at DESC,r.rowid DESC,e.seq DESC LIMIT 80",
                (session_id, buyer_id),
            ) as cursor:
                events = [{"id": f"{item['run_id']}:{item['seq']}", "type": item["type"],
                           "timestamp": item["timestamp"]} for item in await cursor.fetchall()]
            return self.projection.session({"id": session_id, "title": row["title"], "messages": list(messages.values()),
                    "presentations":list(presentations.values()), "processes": processes, "events": events[::-1],
                    "state": json.loads(row["state_json"]), "run": run, "updatedAt": int(row["updated_at"] * 1000)})
