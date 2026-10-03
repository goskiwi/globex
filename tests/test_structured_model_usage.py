"""原生 LangChain 结构化生成，HTTP 传输隔离；失败及取消仍逐次计量。"""
import asyncio
import json
import httpx
import openai
import pytest
from pydantic import BaseModel

from app.infrastructure.context_usage import context_usage_sink, context_call_kind
from app.infrastructure.budget import init_budget
from tests.native_model_helpers import client_model, messages


class Summary(BaseModel):
    goal: str


def response(stream=False, *, valid=True, usage=True):
    tokens = {"prompt_tokens":123,"completion_tokens":17,"total_tokens":140} if usage else None
    call = {"id":"summary","type":"function","function":{"name":"Summary","arguments":'{"goal":"旅行"}'}}
    if not stream:
        return httpx.Response(200,json={"id":"s","object":"chat.completion","created":1,"model":"fixture",
            "choices":[{"index":0,"message": {"role":"assistant","content":None,"tool_calls":[call]}
                if valid else {"role":"assistant","content":"invalid"},"finish_reason":"tool_calls" if valid else "stop"}],
            "usage":tokens})
    chunks = [{"id":"s","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[
        {"index":0,"delta":{"tool_calls":[{"index":0,**call}]} if valid else {"content":"invalid"},"finish_reason":None}]},
        {"id":"s","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[
            {"index":0,"delta":{},"finish_reason":"tool_calls" if valid else "stop"}]},
        {"id":"s","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[],"usage":tokens}]
    return httpx.Response(200,headers={"content-type":"text/event-stream"},
        content="".join("data: "+json.dumps(chunk)+"\n\n" for chunk in chunks)+"data: [DONE]\n\n")


@pytest.mark.parametrize("stream",[False,True])
@pytest.mark.parametrize("outcome",["success","invalid","error","cancel"])
async def test_sdk_structured_request_usage(tmp_path,stream,outcome):
    entered=asyncio.Event()
    async def handler(request):
        body=json.loads(request.content)
        assert body["tools"][0]["function"]["name"]=="Summary"
        if outcome=="error":
            return httpx.Response(503,json={"error":{"message":"connection failed"}})
        if outcome=="cancel":
            entered.set()
            await asyncio.Event().wait()
        return response(stream,valid=outcome=="success")
    model=await client_model(tmp_path,handler,stream=stream)
    samples=[]
    token=context_usage_sink.set(samples.append); kind=context_call_kind.set("summary")
    task=None
    try:
        structured=model.with_structured_output(Summary,method="function_calling")
        if outcome=="cancel":
            task=asyncio.create_task(structured.ainvoke(messages()))
            await asyncio.wait_for(entered.wait(),2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome=="success":
            assert (await structured.ainvoke(messages())).goal=="旅行"
        else:
            with pytest.raises(ValueError if outcome=="invalid" else openai.APIStatusError):
                await structured.ainvoke(messages())
        # 原生 LangChain 不做旧 SDK 的 forced/auto/none 三次隐式解析重试。
        assert len(samples)==1 and samples[0]["kind"]=="summary"
        assert samples[0]["input_tokens"]==(123 if outcome in {"success","invalid"} else None)
        assert samples[0]["output_tokens"]==(17 if outcome in {"success","invalid"} else None)
        assert not model.gateway._semaphore.locked()
    finally:
        context_usage_sink.reset(token);context_call_kind.reset(kind)
        if task:await asyncio.gather(task,return_exceptions=True)
        await model.aclose()


async def test_structured_budget_denial_never_calls_gateway(tmp_path):
    def forbidden(request):
        raise AssertionError("预算不足不能请求上游")
    model=await client_model(tmp_path,forbidden)
    budget=init_budget(10)
    try:
        with pytest.raises(RuntimeError,match="预算不足"):
            await model.with_structured_output(Summary,method="function_calling").ainvoke(messages())
        assert budget.used==budget.reserved==0
    finally:
        init_budget(0);await model.aclose()


async def test_structured_usage_settles_budget_and_schema_is_reserved(tmp_path):
    budget=init_budget(100000)
    def handler(request):
        assert budget.reserved>1024
        assert json.loads(request.content)["max_completion_tokens"]==1024
        return response()
    model=await client_model(tmp_path,handler)
    model.max_tokens=1024
    try:
        await model.with_structured_output(Summary,method="function_calling").ainvoke(messages())
        assert budget.used==140 and budget.reserved==0
    finally:
        init_budget(0);await model.aclose()


async def test_native_compatibility_retry_records_both_requests(tmp_path):
    calls=[];samples=[]
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:
            return httpx.Response(400,json={"error":{"message":"unsupported cache_control"}})
        return response()
    model=await client_model(tmp_path,handler)
    model.cache_mode="explicit"
    token=context_usage_sink.set(samples.append)
    try:
        assert (await model.with_structured_output(Summary,method="function_calling").ainvoke(messages())).goal=="旅行"
        assert len(calls)==2 and [s["input_tokens"] for s in samples]==[None,123]
        assert calls[0]["tools"]==calls[1]["tools"] and calls[0]["tool_choice"]==calls[1]["tool_choice"]
    finally:
        context_usage_sink.reset(token);await model.aclose()


async def test_cancel_during_structured_stream_closes_and_preserves_partial_usage(tmp_path):
    waiting=asyncio.Event();closed=[]
    class Wire(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield ('data: '+json.dumps({"id":"s","object":"chat.completion.chunk","created":1,
                "model":"fixture","choices":[],"usage":{"prompt_tokens":123,"completion_tokens":17,"total_tokens":140}})+'\n\n').encode()
            waiting.set()
            await asyncio.Event().wait()
        async def aclose(self):
            closed.append(True)
    model=await client_model(tmp_path,lambda request:httpx.Response(200,
        headers={"content-type":"text/event-stream"},stream=Wire()),stream=True)
    samples=[];token=context_usage_sink.set(samples.append)
    task=asyncio.create_task(model.with_structured_output(Summary,method="function_calling").ainvoke(messages()))
    try:
        await asyncio.wait_for(waiting.wait(),2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert closed and not model.gateway._semaphore.locked()
        assert len(samples)==1 and samples[0]["input_tokens"]==123
    finally:
        await asyncio.gather(task,return_exceptions=True)
        context_usage_sink.reset(token);await model.aclose()


@pytest.mark.parametrize("stream",[False,True])
async def test_success_without_provider_usage_remains_unknown(tmp_path,stream):
    model=await client_model(tmp_path,lambda request:response(stream,usage=False),stream=stream)
    samples=[];token=context_usage_sink.set(samples.append)
    try:
        result=await model.with_structured_output(Summary,method="function_calling").ainvoke(messages())
        assert result.goal=="旅行"
        assert len(samples)==1 and samples[0]["input_tokens"] is None and samples[0]["output_tokens"] is None
    finally:
        context_usage_sink.reset(token);await model.aclose()
