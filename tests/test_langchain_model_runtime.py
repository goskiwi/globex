"""通过真实 LangChain/OpenAI 解析链验证预算及中断，只替换 HTTP 传输。"""
import asyncio
import json

from openai import AsyncOpenAI
import httpx
import pytest
from langchain_core.messages import HumanMessage

from app.infrastructure.langchain_model import ChatModelAdapter
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.budget import init_budget
from app.application.runtime.errors import ExecutionStopped
from app.infrastructure.operational_metrics import begin_request, finish_request


async def test_streamed_tool_call_has_first_output_time_and_reported_reasoning_usage():
    from app.infrastructure.context_usage import context_usage_sink
    chunks = [
        {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call-probe',
            'type': 'function', 'function': {'name': 'probe', 'arguments': '{}'}}]}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
         'usage': {'prompt_tokens': 12, 'completion_tokens': 10, 'total_tokens': 22,
                   'completion_tokens_details': {'reasoning_tokens': 7}}}]
    wire = ''.join('data: '+json.dumps({'id': 'stream-test', 'object': 'chat.completion.chunk',
        'created': 1, 'model': 'test', **chunk})+'\n\n' for chunk in chunks)+'data: [DONE]\n\n'
    samples = []
    token = context_usage_sink.set(samples.append)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, text=wire, headers={'content-type': 'text/event-stream'}))) as client:
        model = ChatModelAdapter(model_name='test', client=AsyncOpenAI(api_key='test',
            base_url='https://model.test/v1', http_client=client, max_retries=0))
        try:
            reply = await model.bind_tools([{'type':'function','function':{'name':'probe',
                'parameters':{'type':'object','properties':{}}}}]).ainvoke([HumanMessage(content='调用probe')])
            assert reply.tool_calls[0]['name']=='probe'
            sample = samples[0]
            assert 0 <= sample['prompt_cache']['first_chunk_ms'] <= sample['ttft_ms'] <= sample['elapsed_ms']
            assert sample['prompt_cache']['reasoning_tokens']==7
        finally:
            context_usage_sink.reset(token)


def completion(usage=True):
    result = {"id": "chat-test", "object": "chat.completion", "created": 1, "model": "test",
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "完成"},
                           "finish_reason": "stop"}]}
    if usage:
        result["usage"] = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    return result


@pytest.mark.parametrize("with_usage", [True, False])
async def test_http_usage_and_unknown_cost_are_settled(with_usage):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion(with_usage))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ChatModelAdapter(model_name="test", client=AsyncOpenAI(api_key="test", base_url="https://model.test/v1", http_client=client, max_retries=0), streaming=False, max_tokens=128)
        budget = init_budget(10000)
        observation = begin_request()
        try:
            response = await model.ainvoke([HumanMessage(content="你好")])
            summary = finish_request(observation)
            assert response.content == "完成" and len(calls) == 1
            assert budget.reserved == 0
            assert budget.used == 12 if with_usage else budget.used > 12
            assert summary["usage_complete"] is with_usage
            assert summary["model_calls"] == 1
        finally:
            init_budget(0)
            if not observation.finished:
                finish_request(observation, "error")


async def test_exhausted_budget_never_enters_http():
    def handler(request):
        raise AssertionError("预算不足时不应进入 HTTP")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ChatModelAdapter(model_name="test", client=AsyncOpenAI(api_key="test", http_client=client, max_retries=0))
        budget = init_budget(1)
        try:
            with pytest.raises(ExecutionStopped, match="预算不足"):
                await model.ainvoke([HumanMessage(content="你好")])
            assert budget.reserved == 0 and budget.used == 0
        finally:
            init_budget(0)


class BlockedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b""

    async def aclose(self):
        self.closed = True


async def test_cancel_first_token_closes_wire_and_releases_shared_slot():
    wire = BlockedStream()
    def handler(request):
        return httpx.Response(200, stream=wire, headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ChatModelAdapter(model_name="test", client=AsyncOpenAI(api_key="test", http_client=client, max_retries=0), max_tokens=128)
        throttle = GatewayThrottle(1, 0)
        model.gateway = throttle
        budget = init_budget(10000)
        async def consume():
            async for chunk in model.astream([HumanMessage(content="你好")]):
                pass
        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(wire.started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert wire.closed and budget.reserved == 0 and budget.used > 0
            async with asyncio.timeout(1), throttle.slot():
                pass
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            init_budget(0)
