# -*- coding: utf-8 -*-
"""FastAPI 服务入口

路由：
    POST /commerce/intents                 提交买家意图（同步返回最终回复；启用队列时内部入队后等结果）
    POST /commerce/intents/async           提交买家意图（立即返回 task_id，结果走 WS 或轮询）
    GET  /commerce/tasks/{task_id}         查任务状态（queued / running / done / failed）
    WS   /commerce/events                  订阅会话事件流
    GET  /commerce/orders/{order_id}       查询订单（直连 UseCase，不过 Agent）
    POST /commerce/orders/{order_id}/cancel  取消订单（直连 UseCase）
    GET  /health                           健康检查（含依赖连通性与队列深度）

启动：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python -m app.worker          # 启用队列时另起消费进程

同步接口为什么保留：13 case 评测脚本与前端都依赖它直接返回 final_text，
改成纯异步会一次性搞挂回归与前端。削峰由 worker 并发度保证，与接口形态无关。
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text

from app.application.usecases.intents import SubmitIntentInput
from app.composition import Container, build_container
from app.domain.queue.ports.task_queue import IntentTask
from app.presentation.connection import ConnectionManager
from app.presentation.confirmations import register_confirmation_routes, confirmation_error
from app.presentation.ag_ui import register_ag_ui_routes
from app.presentation.buyer_workspace import register_buyer_workspace_routes
from app.presentation.identity import require_buyer, require_session, require_task, require_metrics_reader
from app.infrastructure.tracing import TracingASGIMiddleware, inject_task_context, install_log_correlation
from app.presentation.dto import (
    CancelOrderRequest,
    SubmitIntentRequest,
    SubmitIntentResponse,
)

install_log_correlation()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s request=%(request_id)s task=%(task_id)s session=%(session_id)s trace=%(trace_id)s prompt=%(prompt_version)s %(message)s")

logger = logging.getLogger(__name__)

# 轮数计数器存活时长：比幂等窗口长得多，让一整段会话都能被正确分类
_TURN_COUNTER_TTL_SECONDS = 86400


def build_app() -> FastAPI:
    state: dict = {}

    async def _forward_remote_events(c: Container) -> None:
        """把其他进程（worker）广播的事件转发给本进程的 WS 订阅者。"""
        if c.backplane is None:
            return
        try:
            async for event in c.backplane.listen():
                c.bus.deliver_local(event)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            logger.warning("事件背板监听中断：%s", err)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        c = await build_container()
        state["c"] = c
        application.state.identity_policy = getattr(c, "identity_policy", None)
        application.state.session_store = getattr(c, "session_store", None)
        application.state.session_owner_binding = getattr(getattr(c, "settings", None), "session_owner_binding", True)
        application.state.metrics_reader_buyers = getattr(getattr(c, "settings", None), "metrics_reader_buyers", ())
        state["connections"] = ConnectionManager(c.bus)
        try:
            await c.startup()
            if c.backplane is not None:
                state["forwarder"] = asyncio.create_task(_forward_remote_events(c))
            yield
        finally:
            forwarder = state.pop("forwarder", None)
            if forwarder is not None:
                forwarder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder
            state.pop("c", None)
            await c.shutdown()

    api = FastAPI(title="Globex 跨境电商 Agent", version="0.4.0", lifespan=lifespan)
    api.add_middleware(TracingASGIMiddleware)

    def container() -> Container:
        if "c" not in state:
            raise HTTPException(status_code=503, detail="服务尚未就绪")
        return state["c"]

    settings_origins = build_container_origins()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Trace-ID"],
    )

    # AG-UI 首版在本进程直跑；旧 intents 接口继续使用原有 Redis 队列语义。
    from app.presentation.auth import register_auth_routes
    register_auth_routes(api, lambda: container().settings)
    from app.presentation.product_media import register_product_media_routes
    register_product_media_routes(api)
    from app.presentation.session_management import register_session_management_routes
    register_session_management_routes(api,lambda:container().session_management)
    register_ag_ui_routes(api, lambda: container().orchestrator, lambda: container().confirmations,
                         get_runtime=lambda: container().runs)
    from app.presentation.context_workspace import register_context_routes
    register_context_routes(api, lambda: container().context_service)
    register_confirmation_routes(api, lambda: container().confirmations)
    register_buyer_workspace_routes(api, lambda: container().buyer_workspace)
    from app.presentation.shopping_forms import register_shopping_form_routes
    register_shopping_form_routes(api, lambda: container().shopping_form_store)
    from app.presentation.favorites import register_favorite_routes
    from app.infrastructure.buyer_favorites import BuyerFavoriteStore
    register_favorite_routes(api, lambda: BuyerFavoriteStore(container().settings.data_dir / "buyer_favorites.db"))

    @api.get("/health")
    async def health() -> dict:
        """依赖连通性一并报出，避免"进程活着但存储已挂"被当成健康。"""
        c = container()
        database = "disabled"
        if c.db_engine is not None:
            try:
                async with c.db_engine.connect() as conn:
                    await conn.execute(text("select 1"))
                database = c.db_engine.url.get_backend_name()
            except Exception as err:  # noqa: BLE001
                database = f"error: {err}"
        trade_database = "disabled"
        trade_engine = getattr(c, "trade_db_engine", None)
        if trade_engine is not None:
            try:
                async with trade_engine.connect() as conn:
                    await conn.execute(text("select 1"))
                trade_database = trade_engine.url.get_backend_name()
            except Exception:
                trade_database = "error"
        redis_state = "disabled"
        if c.cache.enabled:
            redis_state = "ok" if await c.cache.ping() else "error"
        ready = not database.startswith("error") and trade_database != "error" and redis_state != "error"
        result = {
            "status": "ok" if ready else "degraded",
            "model": c.settings.llm_model,
            "runtime": getattr(c, "runtime", {}),
            "database": database,
            "trade_database": trade_database,
            "redis": redis_state,
            "semantic_cache": c.semantic_cache.enabled,
            "queue": "enabled" if c.task_queue is not None else "disabled",
            "queue_depth": await c.task_queue.depth() if c.task_queue is not None else 0,
        }
        if getattr(c, "prompt_registry", None) is not None:
            try:
                result["prompt_registry"] = await asyncio.to_thread(c.prompt_registry.describe)
            except ValueError:
                ready = False
                result["status"] = "degraded"
                result["prompt_registry"] = {"status": "unavailable"}
        return result if ready else JSONResponse(status_code=503, content=result)

    @api.get("/internal/metrics", include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        await require_metrics_reader(request)
        from app.infrastructure.operational_metrics import registry
        return Response(registry.prometheus(), media_type="text/plain; version=0.0.4",
                        headers={"Cache-Control": "no-store"})

    @api.get("/internal/metrics/summary", include_in_schema=False)
    async def metrics_summary(request: Request) -> JSONResponse:
        await require_metrics_reader(request)
        from app.infrastructure.operational_metrics import registry
        return JSONResponse({"metrics": registry.snapshot(), "alerts": registry.alerts()},
                            headers={"Cache-Control": "no-store"})

    async def submit_run(request: Request, body: SubmitIntentRequest):
        from app.infrastructure.ag_ui_journal import JournalConflict, JournalForbidden, JournalNotFound
        from app.domain.session.ports.session_store import SessionStoreError
        from app.presentation.identity import session_error
        buyer=await require_buyer(request,body.buyer_id)
        try:
            return await container().runs.submit_intent(buyer_id=buyer,raw_query=body.raw_query,
                session_id=body.shopping_session_id,request_id=body.request_id,locale=body.locale,currency=body.currency)
        except SessionStoreError as error:
            raise session_error(error) from error
        except (JournalConflict,JournalForbidden,JournalNotFound) as error:
            code=403 if isinstance(error,JournalForbidden) else 404 if isinstance(error,JournalNotFound) else 409
            raise HTTPException(code,str(error)) from error
        except ValueError as error:
            raise HTTPException(422,str(error)) from error

    def task_state(run):
        return {'queued':'queued','running':'running','completed':'done'}.get(run['status'],'failed')

    @api.post('/commerce/intents',response_model=SubmitIntentResponse)
    async def submit_intent(request: Request,body: SubmitIntentRequest):
        c=container();run=await submit_run(request,body)
        timeout=c.settings.queue_wait_seconds if c.task_queue is not None else None
        run=await c.runs.wait(run['runId'],body.buyer_id,timeout)
        result=run['result']
        if result is None:
            return SubmitIntentResponse(run_id=run['runId'],shopping_session_id=run['threadId'],final_text='请求已接收，仍在排队或执行；可按运行ID查询。',
                status='partial',stop_reason='wait_timeout')
        return SubmitIntentResponse(run_id=run['runId'],shopping_session_id=run['threadId'],final_text=result['text'],
            status=result['status'],stop_reason=result['stop_reason'])

    @api.post('/commerce/intents/async')
    async def submit_intent_async(request: Request,body: SubmitIntentRequest):
        run=await submit_run(request,body)
        return {'shopping_session_id':run['threadId'],'task_id':run['runId'],'state':task_state(run)}

    @api.get('/commerce/tasks/{task_id}')
    async def get_task(request: Request,task_id: str,buyer_id: str=Query(min_length=1)):
        from app.infrastructure.ag_ui_journal import JournalForbidden,JournalNotFound
        buyer=await require_buyer(request,buyer_id)
        try:run=await container().runs.journal.run(task_id,buyer)
        except (JournalForbidden,JournalNotFound) as error:
            raise HTTPException(403 if isinstance(error,JournalForbidden) else 404,str(error)) from error
        result=run['result']
        return {'task_id':run['runId'],'state':task_state(run),'final_text':result['text'] if result else '',
                'execution_status':result['status'] if result else None,'stop_reason':result['stop_reason'] if result else None,
                'error':run['state'].get('executionError'),'queue_position':None}

    @api.websocket("/commerce/events")
    async def commerce_events(websocket: WebSocket) -> None:
        await state["connections"].serve(websocket)

    @api.get("/commerce/orders")
    async def list_orders(request: Request, buyer_id: str = Query(min_length=1),
                          status: str | None = Query(default=None, pattern="^(CONFIRMED|CANCELLED|DRAFT)$"),
                          offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100)) -> dict:
        buyer = await require_buyer(request, buyer_id)
        return await container().trade_store.list_orders(buyer_id=buyer, status=status, offset=offset, limit=limit)

    @api.get("/commerce/orders/{order_id}")
    async def get_order(request: Request, order_id: str, buyer_id: str = Query(min_length=1)) -> dict:
        buyer_id = await require_buyer(request, buyer_id)
        try:
            return await container().query_order.execute(order_id, buyer_id=buyer_id)
        except ValueError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err

    @api.post("/commerce/orders/{order_id}/cancel")
    async def cancel_order_endpoint(request: Request, order_id: str, body: CancelOrderRequest) -> dict:
        body.buyer_id = await require_buyer(request, body.buyer_id)
        await require_session(request, body.buyer_id, body.session_id, create=True)
        try:
            return await container().cancel_order.execute(order_id, body.reason, buyer_id=body.buyer_id, session_id=body.session_id)
        except ValueError as err:
            raise confirmation_error(err) from err

    return api


def build_container_origins() -> list[str]:
    """CORS 需要在 app 构造期就确定，此处单独读一次配置。"""
    from app.infrastructure.settings import load_settings

    return load_settings().cors_origins


app = build_app()


if __name__ == "__main__":
    import uvicorn

    from app.infrastructure.settings import load_settings

    uvicorn.run(app, host="0.0.0.0", port=load_settings().port)
