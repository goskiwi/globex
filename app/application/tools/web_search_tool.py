# -*- coding: utf-8 -*-
"""web_search_tool

Web 实时资料兜底工具（Tavily HTTP API）：跨境政策、关税规则、清关限制、
商品评测趋势等模型知识覆盖不到的问题走这里。

TAVILY_API_KEY 未配置时组装根不注册本工具（Agent 看不到它）。

函数签名直接用于 LangChain 工具 schema。
"""
import json

import httpx
from app.application.runtime.results import ToolResult, ToolResultState

from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.settings import Settings

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


def build_web_search_tool(settings: Settings, bus: TradeEventBus):
    api_key = settings.tavily_api_key

    async def web_search_tool(query: str, max_results: int = 5) -> ToolResult:
        """联网搜索外部实时资料（跨境政策 / 关税规则 / 清关限制 / 评测趋势）。

        Args:
            query (`str`):
                搜索关键词，如 "美国 800 美元免税额度 最新政策"。
            max_results (`int`):
                返回结果条数，默认 5。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(session_id, "tool.invoke", {"tool": "web_search_tool", "args": {"query": query}})
        # 执行故障交给运行时按异常类型处理，不降成业务拒绝。
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                _TAVILY_ENDPOINT,
                json={"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
            )
            response.raise_for_status()
            body = response.json()
        results = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "content": item.get("content", "")[:500]}
            for item in body.get("results", [])
        ]
        return ToolResult(data={"results": results}, state=ToolResultState.SUCCESS)

    return web_search_tool
