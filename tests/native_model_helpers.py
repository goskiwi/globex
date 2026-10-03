"""原生 LangChain 模型的 HTTP 隔离夹具。"""
from dataclasses import replace
import httpx
from openai import AsyncOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.infrastructure.llm import create_chat_model
from app.infrastructure.throttle import GatewayThrottle
from tests.test_retrieval import _settings
from tests.test_prompt_cache import completion


def messages():
    return [SystemMessage(content="核实商品"), HumanMessage(content="查询商品")]


async def client_model(tmp_path, handler, stream=False):
    config = replace(_settings(tmp_path), llm_api_key="test-key",
                     llm_base_url="https://model.test/v1", llm_fallback_model="", llm_max_retries=0)
    client = AsyncOpenAI(api_key="test-key", base_url=config.llm_base_url,
        max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    model = create_chat_model(config, stream=stream, throttle=GatewayThrottle(1, 0), client=client)
    return model
