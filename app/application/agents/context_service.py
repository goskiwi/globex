# -*- coding: utf-8 -*-
"""LangGraph 会话上下文整理服务。"""
from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import AsyncExitStack

from app.domain.session.ports.session_store import SessionNotFound, StaleSessionWrite
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.application.runtime.context_summary import render_summary


class ContextService:
    def __init__(self, orchestrator, store, evidence, confirmations=None):
        self.orchestrator, self.store, self.evidence = orchestrator, store, evidence
        self.confirmations = confirmations
        self.tasks: dict[str, asyncio.Task] = {}

    async def startup(self):
        if hasattr(self.store, "recover_context_operations"):
            await self.store.recover_context_operations()

    async def shutdown(self):
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    async def view(self, session, buyer):
        result = await self.store.context_view(session, buyer)
        result.pop("working", None)
        result["strategy"] = "langgraph_checkpoint"
        checkpointer = self.orchestrator._sessions._main_factory._checkpointer
        checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": session}})
        if checkpoint:
            values = checkpoint.checkpoint.get("channel_values", {})
            result["summary"] = render_summary(values['context_summary']) if values.get('context_summary') is not None else ''
            result["statistics"] = values.get("context_statistics", {})
            from app.application.agents.shopping_state import ShoppingWork
            if "shopping_work" in values:
                try:
                    result["working"] = ShoppingWork.model_validate(values["shopping_work"]).model_dump()
                except ValueError:
                    result.pop("working", None)
                    result["working_notice"] = "旧购物状态不支持继续执行，请新建会话；历史对话保留。"
        return result

    async def start(self, session, buyer, request_id, expected_revision):
        lock = self.orchestrator._session_locks.setdefault(session, asyncio.Lock())
        identifier = hashlib.sha256(json.dumps([session, buyer, request_id]).encode()).hexdigest()
        try:
            old = await self.store.context_operation(identifier, buyer)
            if old["expected_revision"] != expected_revision:
                raise StaleSessionWrite("相同请求不可换版本")
            return old
        except SessionNotFound:
            pass
        if lock.locked():
            raise StaleSessionWrite("当前正在选购或整理，请等待完成")
        await lock.acquire()
        try:
            if self.confirmations is not None:
                saved = await self.confirmations.list(buyer, session)
                if any(item.get("status") == "pending" and not item.get("expired") for item in saved.get("confirmations", [])):
                    raise StaleSessionWrite("请先完成或拒绝待确认操作")
            operation, created = await self.store.create_context_operation(session, buyer, request_id, expected_revision)
            if not created:
                lock.release()
                return operation
            task = asyncio.create_task(self._run(operation, buyer, lock), name="context:" + operation["operation_id"])
            self.tasks[operation["operation_id"]] = task
            task.add_done_callback(lambda _: self.tasks.pop(operation["operation_id"], None))
            return operation
        except BaseException:
            lock.release()
            raise

    async def _run(self, operation, buyer, lock):
        session_id, identifier = operation["session_id"], operation["operation_id"]
        context_token = ShoppingContext.set(ShoppingContextSnapshot(session_id, buyer, "zh-CN", "CNY"))
        owner = asyncio.current_task()
        async def heartbeat():
            try:
                while True:
                    await asyncio.sleep(5)
                    await self.store.renew_context_operation(identifier)
            except asyncio.CancelledError:
                raise
            except Exception:
                owner.cancel()
        renewal = asyncio.create_task(heartbeat())
        try:
            async with AsyncExitStack() as stack:
                lease_factory = self.orchestrator._session_lease_factory
                lease = await stack.enter_async_context(lease_factory(session_id)) if lease_factory else None
                await self.orchestrator._sessions.invalidate(session_id)
                view = await self.store.context_view(session_id, buyer)
                if view["revision"] != operation["expected_revision"]:
                    raise StaleSessionWrite("选购记录已更新，请重试整理")
                graph_session = await self.orchestrator._sessions.get_or_create(session_id)
                state = await graph_session.graph.aget_state(graph_session.config)
                if state.interrupts:
                    raise StaleSessionWrite("尚有待审批工具")
                policy = graph_session.context_policy
                if policy is None:
                    raise RuntimeError("当前图未提供完整请求装配，不能独立按裸历史整理")
                updates = await policy.compact_checkpoint(state.values, force=True)
                statistics = updates["context_statistics"]
                status = statistics["status"]
                if updates["messages"]:
                    await graph_session.graph.aupdate_state(graph_session.config, updates)
                if lease and not lease.is_valid():
                    raise StaleSessionWrite("整理执行权已失效")
                if not await self.orchestrator._sessions.persist(session_id):
                    raise StaleSessionWrite("会话保存未完成")
                await self.store.finish_context_operation(identifier, buyer, status, {
                    "statistics": statistics,
                    "message": "当前无需整理" if status == "noop" else "上下文已整理，原始运行记录保留",
                })
        except asyncio.CancelledError:
            await self.orchestrator._sessions.invalidate(session_id)
            await asyncio.shield(self.store.finish_context_operation(
                identifier, buyer, "interrupted", {"message": "整理已中断，原记录保留"}
            ))
        except Exception as error:
            await self.orchestrator._sessions.invalidate(session_id)
            await self.store.finish_context_operation(identifier, buyer, "failed", {
                "message": "本次整理未完成，原记录保留；可稍后重试",
                "error_code": type(error).__name__,
            })
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            ShoppingContext.reset(context_token)
            lock.release()
