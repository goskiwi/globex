"""真实模型解释合成多轮需求，经原生图和真实检索工具检查执行条件。"""
import asyncio
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.tools.shopping_state_tool import build_shopping_state_tool
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.application.agents.shopping_state import ShoppingWork, compile_search
from app.application.runtime.working_state import WorkingStateMiddleware
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.buyer.preference import BuyerPreference, MaterialExclusion
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.settings import load_settings
from app.infrastructure.context_usage import context_usage_sink
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository


async def main():
    with tempfile.TemporaryDirectory(prefix="globex-state-") as folder:
        settings = replace(load_settings(), data_dir=Path(folder), llm_fallback_model="", llm_max_retries=0)
        model = create_chat_model(settings, client=create_chat_client(settings))
        prefs = (BuyerPreference("synthetic", "dislike", "不要合成聚合物材质",constraint=MaterialExclusion(("合成聚合物",)),evidence="不要合成聚合物材质"),)
        token = ShoppingContext.set(ShoppingContextSnapshot("state-live", "synthetic", "zh-CN", "CNY", prefs))
        bus = TradeEventBus()
        store = ContextEvidenceStore(Path(folder)/'evidence.db')
        fn = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus, store)
        from app.application.prompts.loader import load_prompts
        graph = create_agent(model, tools=[as_langchain_tool(fn),build_shopping_state_tool(store)],
            system_prompt=load_prompts()['main_agent']['system_prompt'],
            middleware=[WorkingStateMiddleware()], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "state-live"}, "recursion_limit": 16}
        prompts = ["预算300元，帮我找背包，寄到中国。", "预算不限了，其他不变。",
                   "背包不要超过500元。", "这次背包可以接受塑料，其他不变。",
                   "这次还是恢复不要塑料的偏好，其他不变。",
                   "背包预算是多少？只回顾，不修改要求。", "重新开始找背包，预算300元。"]
        previous = None
        measurements=[]
        usage_token=context_usage_sink.set(measurements.append)
        print(json.dumps({'model':settings.llm_model,'base_url':settings.llm_base_url,'scope':'synthetic data only'}),flush=True)
        try:
            for index, text in enumerate(prompts):
                events = []
                started = time.perf_counter()
                print(json.dumps({'round':index+1,'phase':'started'}),flush=True)
                with observe_run_events(lambda e: events.append(e.payload) if e.type == "tool.invoke" else None):
                    async with asyncio.timeout(90):
                        result = await graph.ainvoke({"messages": [HumanMessage(name="synthetic", content=text)]}, config)
                work = ShoppingWork.model_validate(result["shopping_work"])
                compiled = compile_search(work, prefs)
                cap = [300, None, 500, 500, 500, 500, 300][index]
                passed = compiled["parameters"]["price_max_major"] == cap
                if index == 3:
                    passed = passed and compiled["parameters"]["excluded_material_tags"] == []
                if index in (4,5,6):
                    passed = passed and compiled["parameters"]["excluded_material_tags"] == ["合成聚合物"]
                if index == 5:
                    passed = passed and work.filters == previous.filters and not events
                elif not events:
                    passed = False
                print(json.dumps({"round": index + 1, "input": text, "passed": bool(passed),
                    "elapsed_s": round(time.perf_counter() - started, 3), "work": work.model_dump(),
                    "tool_requests": events, "answer": result["messages"][-1].content}, ensure_ascii=False), flush=True)
                if not passed:
                    raise SystemExit(1)
                previous = work
        except Exception as error:
            print(json.dumps({"passed": False, "error_type": type(error).__name__,
                'elapsed_s':round(time.perf_counter()-started,3),'usage':measurements}), flush=True)
            raise SystemExit(1) from None
        finally:
            ShoppingContext.reset(token)
            context_usage_sink.reset(usage_token)
            await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
