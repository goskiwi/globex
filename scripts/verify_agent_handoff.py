"""真实模型验证子任务交接；仅合成只读工具，不接入买家、订单或真实目录。"""
import asyncio
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from langchain_core.tools import tool

from app.application.agents.search_agent import SearchAgentFactory
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.settings import load_settings
from app.infrastructure.llm import create_chat_client
from app.infrastructure.throttle import GatewayThrottle
from app.application.harness.loop_detector import LoopDetector


async def main():
    with tempfile.TemporaryDirectory(prefix="globex-handoff-live-") as folder:
        settings = replace(load_settings(), data_dir=Path(folder), harness_enabled=True,
                           llm_fallback_model="", llm_max_retries=0)
        async with create_chat_client(settings) as model_client:
            bus, calls = TradeEventBus(), []
    
            @tool
            def product_search_tool(normalized_query: str = "背包") -> dict:
                """检索三个合成背包，按价格上限过滤，返回价格重量与 SKU；没有其他商品属性。"""
                calls.append("product_search_tool")
                price_max_major = ShoppingContext.current().effective_search["parameters"]["price_max_major"]
                data = {"tool": "product_search_tool", "result_ref": "ctx_live_demo",
                    "recall_strategy": "synthetic_fixture", "hits": [
                    {"product_id": f"P{1001 + i}", "title": f"合成背包{i + 1}", "price_major": price,
                     "currency": "CNY", "weight_kg": weight, "skus": [{"sku_id": f"P{1001 + i}-S1"}]}
                    for i, (price, weight) in enumerate([(240, 0.6), (290, 0.9), (280, 0.8)])
                    if price_max_major is None or price <= price_max_major]}
                bus.publish("handoff-live", "tool.result", data)
                return data
    
            factory = SearchAgentFactory(settings, None, bus, None, CircuitBreakerRegistry(), GatewayThrottle(3, 0), model_client=model_client)
            factory.bind_harness(LoopDetector())
            factory.build_tools = lambda: [product_search_tool]
            build = factory.build
            terminal = {}
    
            def observed_build():
                graph = build()
    
                async def invoke(inputs, **kwargs):
                    result = await graph.ainvoke(inputs, config={"recursion_limit": 30}, **kwargs)
                    last = result["messages"][-1]
                    terminal.update(type=last.type, name=last.name, content=last.content,
                                    result=result.get("handoff_result"), attempts=result.get("handoff_attempts", 0))
                    return result
    
                return SimpleNamespace(ainvoke=invoke)
    
            factory.build = observed_build
            dispatch = build_task_dispatch_tool(factory, factory, bus)
            token = ShoppingContext.set(ShoppingContextSnapshot("handoff-live", "synthetic-buyer", "zh-CN", "CNY"))
            started = time.perf_counter()
            passed = False
            try:
                async with asyncio.timeout(60):
                    output = await dispatch("search_agent", {
                        "goal": "读取演示候选，选择300元以内的轻便背包。",
                        "filters": {"price_max_major": 300, "target_currency": "CNY"},
                    })
                result = output.data
                passed = (result["status"] == "completed" and bool(result["candidates"])
                          and result["candidates"][0]["product_id"] == "P1001" and bool(calls))
                report = {"model": settings.llm_model, "passed": passed, "calls": calls, "result": result,
                          "harness_enabled": settings.harness_enabled,
                          "synthetic_terminal_message": terminal,
                          "scope": "真实供应商；生产 Search 工厂及派发；合成只读工具；无真实买家、目录或交易"}
            except Exception as error:
                report = {"passed": False, "error_type": type(error).__name__}
            finally:
                ShoppingContext.reset(token)
            report["elapsed_s"] = round(time.perf_counter() - started, 3)
            print(json.dumps(report, ensure_ascii=False))
            if not passed:
                raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
