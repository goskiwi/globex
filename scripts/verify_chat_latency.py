"""使用项目模型工厂测量合成文本及只读工具往返，不读取买家数据。"""
import asyncio
import argparse
import json
import time

from langchain.agents import create_agent
from langchain_core.tools import tool

from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.settings import load_settings


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


async def main(disable_thinking=False):
    settings = load_settings()
    async with create_chat_client(settings) as model_client:
        emit({"kind": "config", "model": settings.llm_model,
              "base_url": settings.llm_base_url, "fallback": settings.llm_fallback_model,
              "thinking": "disabled" if disable_thinking else "provider_default",
              "scope": "synthetic prompts; project model factory; no catalog or buyer data"})
        prompt = "合成测速：预算300元，优先轻便。背包A：240元、0.6kg、20L；B：290元、0.9kg、28L。用80到120个汉字推荐一款并说明取舍。"
        for trial in range(1, 4):
            model = create_chat_model(settings, client=model_client)
            if disable_thinking:
                model = model.bind(extra_body={"thinking": {"type": "disabled"}})
            started = time.perf_counter()
            first_text = None
            aggregate = None
            try:
                async with asyncio.timeout(55):
                    async for chunk in model.astream(prompt):
                        if chunk.content and first_text is None:
                            first_text = time.perf_counter() - started
                        aggregate = chunk if aggregate is None else aggregate + chunk
                emit({"kind": "text", "trial": trial, "first_text_s": first_text,
                      "total_s": time.perf_counter() - started,
                      "usage": aggregate.usage_metadata if aggregate else None,
                      "metadata": aggregate.response_metadata if aggregate else None,
                      "text": aggregate.content if aggregate else ""})
                if not aggregate or not aggregate.content:
                    raise SystemExit("模型未返回可见文本")
            except Exception as error:
                emit({"kind": "text", "trial": trial, "error_type": type(error).__name__,
                      "total_s": time.perf_counter() - started})
                raise SystemExit(1) from None
    
        calls = []
    
        @tool
        def lookup_demo_product(product_id: str) -> dict:
            """查询合成商品A的当前价格与库存；只读演示数据。"""
            calls.append(product_id)
            return {"product_id": "A", "price_cny": 240, "stock": 3}
    
        graph_model = create_chat_model(settings, client=model_client)
        if disable_thinking:
            graph_model = graph_model.bind(extra_body={"thinking": {"type": "disabled"}})
        graph = create_agent(graph_model, tools=[lookup_demo_product],
                             system_prompt="必须先调用工具核实商品，再用一句中文回答价格与库存。")
        started = time.perf_counter()
        try:
            async with asyncio.timeout(55):
                result = await graph.ainvoke(
                    {"messages": [{"role": "user", "content": "查询商品A现在的价格和库存。"}]},
                    config={"recursion_limit": 8},
                )
            messages = result["messages"]
            emit({"kind": "graph_tool_roundtrip", "total_s": time.perf_counter() - started,
                  "tool_calls": calls, "message_types": [m.type for m in messages],
                  "answer": messages[-1].content,
                  "passed": calls == ["A"] and messages[-1].type == "ai" and bool(messages[-1].content)})
            if calls != ["A"] or messages[-1].type != "ai" or not messages[-1].content:
                raise SystemExit("工具往返验证未通过")
        except Exception as error:
            emit({"kind": "graph_tool_roundtrip", "total_s": time.perf_counter() - started,
                  "error_type": type(error).__name__, "tool_calls": calls})
            raise SystemExit(1) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-thinking", action="store_true", help="仅测试支持此参数的供应商")
    asyncio.run(main(parser.parse_args().disable_thinking))
