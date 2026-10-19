# -*- coding: utf-8 -*-
"""运行独立于 HTTP 订阅存在；日志写入成功后事件才对 SSE 可见。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
import uuid

from ag_ui.core import RunAgentInput
from pydantic import BaseModel, ConfigDict, Field

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.usecases.intents import SubmitIntentInput
from app.application.runtime.execution import ExecutionResult
from app.domain.queue.ports.task_queue import IntentTask, NonRetryableTaskError
from app.infrastructure.queue.redis_stream_queue import current_execution_lease, QueueLeaseLost
from app.infrastructure.tracing import inject_task_context
from app.infrastructure.ag_ui_journal import JournalConflict, JournalForbidden, JournalNotFound
from app.infrastructure.ag_ui_journal import AGUIJournal, JournalLeaseLost
from app.infrastructure.eventbus import observe_run_events

logger = logging.getLogger(__name__)


class RunProperties(BaseModel):
    """入口只接受当前身份与地区字段；调度信息由服务端注入。"""
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    buyerId: str = Field(min_length=1)
    locale: str = 'zh-CN'
    currency: str = Field(default='CNY', pattern=r'^[A-Z]{3}$')


@dataclass
class Running:
    owner: str
    deadline: float
    valid: bool = True
    reason: str = ""
    task: asyncio.Task | None = None
    execution_guard: object = None

    def is_valid(self):
        return self.valid and time.monotonic() < self.deadline and (self.execution_guard is None or self.execution_guard())


class RunService:
    def __init__(self, journal: AGUIJournal, orchestrator, confirmations=None, *, shopping_forms=None, task_queue=None, session_store=None, priority=None, lease_seconds=30, heartbeat_seconds=5):
        self.journal, self.orchestrator, self.confirmations = journal, orchestrator, confirmations
        self.lease_seconds, self.heartbeat_seconds = lease_seconds, heartbeat_seconds
        self.running: dict[str, Running] = {}
        self.shopping_forms = shopping_forms
        self.task_queue, self.session_store, self.priority = task_queue, session_store, priority

    async def startup(self):
        await self.journal.initialize()
        await self.journal.recover_expired()
        if self.task_queue is not None:
            for run in await self.journal.queued():
                await self._enqueue(run)

    async def shutdown(self):
        active = list(self.running.items())
        for _, entry in active:
            entry.reason = "restart"
            entry.valid = False
            if entry.task:
                entry.task.cancel()
        await asyncio.gather(*(entry.task for _, entry in active if entry.task), return_exceptions=True)
        for run_id, entry in active:
            await self.journal.end_owned(run_id, entry.owner)
            self.running.pop(run_id, None)

    @staticmethod
    def _intent(body):
        props = body.forwarded_props or {}
        confirmations = tuple({'interrupt_id': entry.interrupt_id,
                               'approved': False if entry.status == 'cancelled' else entry.payload['approved']}
                              for entry in body.resume or [])
        return SubmitIntentInput(body.thread_id, props['buyerId'], str(props.get('locale','zh-CN')),
            str(props.get('currency','CNY')), body.messages[-1].content.strip(), confirmations,
            body.messages[-1].id, body.run_id)

    async def start(self, body: RunAgentInput, buyer_id: str):
        return await self._admit(body.model_dump(mode='json',by_alias=True), buyer_id)

    async def submit_intent(self, *, buyer_id, raw_query, session_id=None, request_id=None,
                            locale='zh-CN', currency='CNY'):
        if not raw_query.strip():
            raise ValueError('需求不能为空')
        body={'threadId':session_id, 'runId':'run-'+uuid.uuid4().hex, 'state':{},
              'messages':[{'id':'user-'+uuid.uuid4().hex,'role':'user','content':raw_query.strip()}],
              'tools':[], 'context':[], 'forwardedProps':{'buyerId':buyer_id,'locale':locale,'currency':currency}}
        return await self._admit(body,buyer_id,request_id=request_id)

    async def _admit(self, body, buyer_id, *, request_id=None):
        properties=RunProperties.model_validate(body.get('forwardedProps') or {})
        if properties.buyerId != buyer_id:
            raise JournalForbidden('运行身份与请求买家不一致')
        session_id=body['threadId']
        if self.session_store is not None and session_id is not None:
            await self.session_store.assert_owner(session_id,buyer_id,create=True)
        if body['runId'].startswith('form-run-'):
            form=await self.shopping_forms.submitted_run(buyer_id,session_id,body['runId']) if self.shopping_forms else None
            submission=form.get('submission') if form else None
            if not submission or submission['query']!=body['messages'][-1]['content'] or body['messages'][-1]['id']!=body['runId']+':user':
                raise JournalConflict('继续选购必须使用本次已保存的表单提交')
        if self.task_queue is not None:
            props={**body['forwardedProps'], '_execution':inject_task_context(session_id=session_id or '',task_id=body['runId']),
                   '_priority':await self.priority(session_id) if self.priority is not None else 0}
            body={**body,'forwardedProps':props}
        owner=uuid.uuid4().hex; started=time.monotonic()
        run,created=await self.journal.reserve(body,buyer_id,owner,self.lease_seconds,
                                               queued=self.task_queue is not None,request_id=request_id)
        if self.session_store is not None:
            await self.session_store.assert_owner(run['threadId'],buyer_id,create=True)
            await self.session_store.bind_task_owner(run['runId'],run['threadId'],buyer_id)
        if self.task_queue is not None and run['status']=='queued':
            await self._enqueue(run)
        elif created:
            effective=RunAgentInput.model_validate(run['input'])
            entry=Running(owner,started+self.lease_seconds)
            self.running[run['runId']]=entry
            entry.task=asyncio.create_task(self._produce(effective,self._intent(effective),entry),name='run:'+run['runId'])
        return run

    async def _enqueue(self, run):
        body=RunAgentInput.model_validate(run['input']); intent=self._intent(body)
        if self.session_store is not None:
            await self.session_store.assert_owner(body.thread_id,intent.buyer_id,create=True)
            await self.session_store.bind_task_owner(body.run_id,body.thread_id,intent.buyer_id)
        props=body.forwarded_props or {}
        await self.task_queue.enqueue(IntentTask(body.run_id,body.thread_id,intent.buyer_id,intent.locale,intent.currency,
            intent.raw_query,priority=props.get('_priority',0),**props.get('_execution',{})))

    async def execute_queued(self, task):
        lease=current_execution_lease()
        if lease is None or not lease.is_valid():
            raise RuntimeError('worker未持有有效执行租约')
        owner=uuid.uuid4().hex; started=time.monotonic()
        try:
            run,claimed=await self.journal.claim(task.task_id,task.buyer_id,owner,self.lease_seconds)
        except (JournalNotFound,JournalForbidden) as error:
            raise NonRetryableTaskError('任务没有可接纳的持久运行') from error
        if run['threadId']!=task.shopping_session_id:
            raise NonRetryableTaskError('排队任务与持久运行会话不一致')
        if claimed:
            body=RunAgentInput.model_validate(run['input'])
            entry=Running(owner,started+self.lease_seconds,execution_guard=lease.is_valid)
            self.running[body.run_id]=entry
            entry.task=asyncio.create_task(self._produce(body,self._intent(body),entry,started=run['cursor']>0))
            await entry.task
            run=await self.journal.run(body.run_id,task.buyer_id)
        elif run['status']=='running':
            raise QueueLeaseLost('运行仍由另一执行者持有，不重放意图')
        result=run['result']
        if result is None:
            return ExecutionResult('运行结果尚未取得完整持久记录，请先核查状态。','failed','persistence_error')
        return ExecutionResult(result['text'],result['status'],result['stop_reason'],tuple(result['presentations']))

    async def wait(self, run_id, buyer_id, timeout=None):
        deadline=time.monotonic()+timeout if timeout is not None else None
        while True:
            run=await self.journal.run(run_id,buyer_id)
            if run['status'] not in {'queued','running'} or deadline is not None and time.monotonic()>=deadline:
                return run
            await asyncio.sleep(.05)

    async def cancel(self, run_id, buyer_id):
        run = await self.journal.request_stop(run_id, buyer_id)
        entry = self.running.get(run_id)
        if entry and entry.task:
            entry.reason = "stop"
            entry.valid = False
            entry.task.cancel()
            await asyncio.shield(asyncio.gather(entry.task, return_exceptions=True))
            await self.journal.end_owned(run_id, entry.owner, stopped=True)
            self.running.pop(run_id, None)
            run = await self.journal.run(run_id, buyer_id)
        return run

    async def _produce(self, body, intent, entry, *, started=False):
        queue = asyncio.Queue()
        adapter = AGUIRunAdapter(body, queue.put_nowait, authoritative_state=body.state)

        async def write_events():
            while True:
                event = await queue.get()
                if event is None:
                    return
                events = [event]
                ending = False
                while len(events) < 64 and not queue.empty():
                    following = queue.get_nowait()
                    if following is None:
                        ending = True
                        break
                    events.append(following)
                payloads = [value.model_dump(mode="json", by_alias=True, exclude_none=True) for value in events]
                if entry.reason == "restart":
                    for value in payloads:
                        if value["type"] == "RUN_ERROR":
                            value["code"] = "SERVER_RESTART"
                try:
                    await self.journal.append(body.run_id, entry.owner, payloads)
                except Exception:
                    entry.valid = False
                    entry.reason = "restart"
                    if entry.task:
                        entry.task.cancel()
                    raise
                if ending:
                    return

        async def heartbeat():
            while True:
                await asyncio.sleep(self.heartbeat_seconds)
                started = time.monotonic()
                try:
                    if not entry.is_valid():
                        raise JournalLeaseLost('执行租约已失效')
                    renewed = await asyncio.wait_for(self.journal.renew(body.run_id, entry.owner, self.lease_seconds),
                                                    max(0.001, entry.deadline - started))
                    if not renewed:
                        run = await self.journal.run(body.run_id, intent.buyer_id)
                        entry.reason = "stop" if run["stopRequested"] else "restart"
                        raise JournalLeaseLost("运行日志租约失效或收到跨实例停止请求")
                    entry.deadline = started + self.lease_seconds
                except Exception:
                    entry.valid = False
                    if entry.task:
                        entry.task.cancel()
                    return

        writer = asyncio.create_task(write_events())
        lease_task = asyncio.create_task(heartbeat())
        adapter.snapshot() if started else adapter.start()
        try:
            if self.confirmations is not None:
                saved = await self.confirmations.restore(intent.buyer_id, intent.shopping_session_id)
                adapter.state["confirmations"] = saved["confirmations"]
                adapter.snapshot()
            with observe_run_events(adapter.on_trade_event):
                options={'event_observer':adapter.on_agent_event,'persistence_guard':entry.is_valid}
                if entry.execution_guard is not None:
                    options['fresh_session']=True
                result=await self.orchestrator.handle_intent(intent,**options)
            if not entry.is_valid():
                raise asyncio.CancelledError()
            if result.error or adapter.error:
                adapter.fail(result.error if getattr(result,"error_code",None) in {"SESSION_VERSION_CHANGED","CONTEXT_CAPACITY_EXCEEDED","PERSISTENCE_FAILED"} else adapter.error or "本轮服务暂时未能完成请求，请重试。", code=getattr(result,"error_code",None))
            else:
                adapter.finish(result.final_text, result.status, result.stop_reason)
        except asyncio.CancelledError:
            adapter.fail("本轮已明确停止" if entry.reason == "stop" else "服务重启或执行中断，已保存的内容仍可恢复。", cancelled=entry.reason == "stop")
        except Exception:
            logger.exception("AG-UI 运行失败：%s", body.run_id)
            adapter.fail("本轮服务暂时未能完成请求，请重试。")
        finally:
            lease_task.cancel()
            await asyncio.gather(lease_task, return_exceptions=True)
            queue.put_nowait(None)
            try:
                await writer
            except Exception:
                logger.exception("AG-UI 日志写入失败，等待持久租约过期后标记中断：%s", body.run_id)
            entry.valid = False
            try:
                await self.journal.end_owned(body.run_id,entry.owner,stopped=entry.reason=='stop')
            except Exception:
                logger.exception('运行终态保存失败，等待持久租约恢复：%s',body.run_id)
            self.running.pop(body.run_id, None)


async def queue_priority(settings, cache, session_id):
    """保留当前长会话低优先级策略；故障时不阻断运行接收。"""
    if not settings.queue_priority_enabled or not cache.enabled:
        return 0
    try:
        key=f'globex:turns:{session_id}'
        current=int(await cache.get_raw(key) or 0)+1
        await cache.set_json(key,current,86400)
        return int(current>=settings.queue_large_request_turns)
    except Exception:
        return 0
