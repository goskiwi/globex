"""原生 LangChain/OpenAI HTTP 请求、流取消与资源释放。"""
import asyncio
import json

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler


class TextCollector(AsyncCallbackHandler):
    def __init__(self):
        self.parts = []

    async def on_llm_new_token(self, token, **kwargs):
        self.parts.append(token)


from app.infrastructure.model_protocol import ModelProtocolViolation
from tests.native_model_helpers import client_model, messages, completion


class DraftWire(httpx.AsyncByteStream):
    def __init__(self):
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield ("data: " + json.dumps({"id":"s","object":"chat.completion.chunk","created":1,
            "model":"fixture","choices":[{"index":0,"delta":{"content":"草稿"},"finish_reason":None}]}) + "\n\n").encode()
        self.waiting.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed.set()


async def test_real_nonstream_response_does_not_probe_missing_attributes_and_releases_slot(tmp_path):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        value = completion()
        value["choices"][0]["message"]["content"] = str(count)
        return httpx.Response(200,json=value)
    model = await client_model(tmp_path,handler)
    try:
        assert (await model.ainvoke(messages())).content == "1"
        assert (await asyncio.wait_for(model.ainvoke(messages()),1)).content == "2"
        assert not model.gateway._semaphore.locked()
    finally:
        await model.aclose()


async def test_request_cancellation_swallowed_by_sdk_is_restored_and_slot_released(tmp_path):
    started=asyncio.Event(); count=0
    async def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            await asyncio.Event().wait()
        return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    task=asyncio.create_task(model.ainvoke(messages()))
    try:
        await asyncio.wait_for(started.wait(),1)
        task.cancel("用户停止，timeout 不应触发重试")
        with pytest.raises(asyncio.CancelledError):await task
        assert count==1
        assert (await asyncio.wait_for(model.ainvoke(messages()),1)).content=="OK"
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        await model.aclose()


async def test_stream_cancellation_does_not_yield_fake_final_response_and_releases_slot(tmp_path):
    wire=DraftWire()
    model=await client_model(tmp_path,lambda request:httpx.Response(200,
        headers={"content-type":"text/event-stream"},stream=wire),stream=True)
    received=TextCollector()
    task=asyncio.create_task(model.ainvoke(messages(), config={"callbacks":[received]}))
    try:
        await asyncio.wait_for(wire.waiting.wait(),1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert wire.closed.is_set() and "".join(received.parts)=="草稿"
        assert task.cancelled(), "只能有文本增量，不能把半成品响应交给图执行"
        async with asyncio.timeout(1), model.gateway.slot():
            pass
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        await model.aclose()


async def test_explicit_interrupted_response_cannot_be_treated_as_success(tmp_path):
    raw=completion();raw["choices"][0]["finish_reason"]="interrupted"
    replies=[raw,completion()]
    model=await client_model(tmp_path,lambda request:httpx.Response(200,json=replies.pop(0)))
    try:
        with pytest.raises(ModelProtocolViolation,match="工具权限"):
            await model.ainvoke(messages())
        assert (await model.ainvoke(messages())).content=="OK"
    finally:
        await model.aclose()


async def test_cancel_message_matching_transient_error_is_never_retried_or_fallback(tmp_path):
    calls=[];started=asyncio.Event()
    async def handler(request):
        calls.append(request);started.set()
        await asyncio.Event().wait()
    model=await client_model(tmp_path,handler)
    task=asyncio.create_task(model.ainvoke(messages()))
    try:
        await asyncio.wait_for(started.wait(),1)
        task.cancel("timeout 429")
        with pytest.raises(asyncio.CancelledError):await task
        assert len(calls)==1 and not model.gateway._semaphore.locked()
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        await model.aclose()


async def test_closing_started_stream_closes_upstream_before_releasing_slot(tmp_path):
    wire=DraftWire()
    model=await client_model(tmp_path,lambda request:httpx.Response(200,
        headers={"content-type":"text/event-stream"},stream=wire),stream=True)
    received=TextCollector()
    task=asyncio.create_task(model.ainvoke(messages(), config={"callbacks":[received]}))
    try:
        await asyncio.wait_for(wire.waiting.wait(), 1)
        assert received.parts == ["草稿"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.closed.is_set() and not model.gateway._semaphore.locked()
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        await model.aclose()
