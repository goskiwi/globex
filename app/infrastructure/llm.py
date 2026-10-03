# -*- coding: utf-8 -*-
"""唯一模型工厂：原生 LangChain 模型，调用治理由 ChatModelAdapter 执行。"""
from __future__ import annotations

from typing import Any
from app.infrastructure.langchain_model import ChatModelAdapter
from openai import AsyncOpenAI
from app.infrastructure.settings import Settings
from app.infrastructure.throttle import GatewayThrottle


def create_chat_client(settings: Settings) -> AsyncOpenAI:
    """调用入口持有此客户端，并负责在退出时关闭；Agent 不创建客户端。"""
    return AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, max_retries=0, timeout=60)


def create_chat_model(settings: Settings, *, client: Any, stream: bool = True,
                      throttle: Any = None, bus: Any = None) -> ChatModelAdapter:
    """预算/限流在单次调用边界，重试/备用模型在 GatewayModelMiddleware。"""
    if settings.prompt_cache_mode not in {"passthrough", "explicit"}:
        raise ValueError("PROMPT_CACHE_MODE 仅支持 passthrough/explicit")
    if settings.prompt_cache_policy not in {"static", "static_history"}:
        raise ValueError("PROMPT_CACHE_POLICY 仅支持 static/static_history")
    if client is None:
        raise ValueError("模型客户端必须由调用入口显式提供")
    model = ChatModelAdapter(
        model_name=settings.llm_model, streaming=stream,
        client=client,
        max_tokens=min(8192,settings.reply_token_budget) if settings.reply_token_budget>0 else 8192,
    )
    model.gateway = throttle or GatewayThrottle(
        settings.llm_max_concurrency, settings.llm_min_interval_seconds)
    model.cache_mode = settings.prompt_cache_mode
    model.cache_policy = settings.prompt_cache_policy
    return model
