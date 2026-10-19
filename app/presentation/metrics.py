"""API 与 Worker 共用指标输出，Prometheus 按目标采集后聚合。"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.presentation.identity import require_metrics_reader


def configure_metrics_access(api, settings, identity_policy):
    api.state.identity_policy = identity_policy
    api.state.metrics_reader_buyers = settings.metrics_reader_buyers
    token = ""
    if settings.metrics_scrape_token_file:
        token = Path(settings.metrics_scrape_token_file).read_text(encoding="utf-8").strip()
        if len(token) < 32 or any(c.isspace() for c in token):
            raise ValueError("指标采集凭据至少需要32个非空白字符")
    api.state.metrics_scrape_token = token


def register_metrics_routes(api: FastAPI, queue_provider=None):
    @api.get("/internal/metrics", include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        await require_metrics_reader(request)
        from app.infrastructure.operational_metrics import registry
        output = registry.prometheus()
        if queue_provider is not None:
            queue = queue_provider()
            values = await queue.metrics() if queue is not None else {
                "depth": 0, "lag": 0, "pending": 0, "dead_letters": 0}
            for name, value in values.items():
                output += f"# TYPE globex_queue_{name} gauge\nglobex_queue_{name} {value}\n"
        return Response(output, media_type="text/plain; version=0.0.4",
                        headers={"Cache-Control": "no-store"})

    @api.get("/internal/metrics/summary", include_in_schema=False)
    async def metrics_summary(request: Request) -> JSONResponse:
        await require_metrics_reader(request)
        from app.infrastructure.operational_metrics import registry
        return JSONResponse({"metrics": registry.snapshot(), "alerts": registry.alerts()},
                            headers={"Cache-Control": "no-store"})
