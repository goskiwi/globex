"""真实模型推荐链验收：合成买家、临时证据库、内置目录，不执行真实交易。"""
import argparse
import asyncio
import json
import tempfile
import time
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.agents.main_agent import MainAgentFactory
from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.agents.search_agent import SearchAgentFactory
from app.application.agents.trade_agent import TradeAgentFactory
from app.application.harness.loop_detector import LoopDetector
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.settings import load_settings
from app.infrastructure.throttle import GatewayThrottle


async def main(comparison=False):
    with tempfile.TemporaryDirectory(prefix="globex-pricing-live-") as folder:
        settings = replace(load_settings(), data_dir=Path(folder), llm_fallback_model="", llm_max_retries=0, harness_enabled=True)
        bus, circuit, throttle = TradeEventBus(), CircuitBreakerRegistry(), GatewayThrottle(3, 0)
        catalog = CatalogSearchUseCase(InMemoryProductRepository())
        model = create_chat_model(settings, stream=False, client=create_chat_client(settings))
        model_client=model.client
        search = SearchAgentFactory(settings, catalog, bus, None, circuit, throttle, model_client=model_client)
        trade = TradeAgentFactory(settings, None, None, None, bus, circuit, throttle, model_client=model_client)
        factory = MainAgentFactory(settings, search, trade, bus, JsonFilePreferenceStore(Path(folder)/"preferences"),
            circuit, throttle, checkpointer=InMemorySaver(), loop_detector=LoopDetector(), model_client=model_client)
        token = ShoppingContext.set(ShoppingContextSnapshot("pricing-live", "synthetic", "zh-CN", "CNY"))
        started = time.monotonic()
        events = []
        try:
            with patch("app.application.agents.main_agent.create_chat_model", return_value=model), observe_run_events(events.append):
                graph = factory.build().graph
                runner = MainAgentOrchestrator.__new__(MainAgentOrchestrator)
                runner._bus = bus
                runner._native_observer = ContextVar("pricing-live-observer", default=None)
                session = SimpleNamespace(graph=graph, config={"configurable": {"thread_id": "pricing-live"}})
                async with asyncio.timeout(90):
                    answer = await runner._reply("pricing-live", session, [HumanMessage(name="synthetic",
                        content=("比较 P1001-S1 和 P1001-S2 两种规格，各买两件寄中国，到手预算450元人民币。给我比较表和主要取舍，不要下单。" if comparison else "想买两件旅行三件套，寄中国，到手总预算450元人民币。请核对P1001-S1是否合适，给出推荐和两件完整费用，不要下单。"))])
            results = [e.payload for e in events if e.type == ("comparison.result" if comparison else "recommendation.result")]
            quote = results[-1]["hits"][0]["landed_price"] if results else None
            passed = bool(quote and quote["items"][0]["sku_id"] == "P1001-S1" and
                          quote["items"][0]["quantity"] == 2 and quote["total_amount_minor"] == 41800
                          and answer.product_delivery_complete and answer.text == results[-1]['guidance'])
            if comparison:
                passed = passed and [c["default_sku_id"] for c in results[-1]["hits"]] == ["P1001-S1", "P1001-S2"]
                passed = passed and results[-1]["hits"][1]["landed_price"]["total_amount_minor"] == 43800
            print(json.dumps({"scenario": "comparison" if comparison else "recommendation","passed": passed, "model": settings.llm_model, "elapsed_s": round(time.monotonic()-started, 3),
                "answer": answer.text, "execution_status": answer.status, "recommendations": results,
                "scope": "真实模型与生产 Main 工厂；合成买家、临时证据库；无支付或真实交易"}, ensure_ascii=False))
            if not passed:
                raise SystemExit(1)
        finally:
            ShoppingContext.reset(token)
            await model.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="store_true")
    asyncio.run(main(parser.parse_args().comparison))
