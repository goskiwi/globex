"""显式缓存通过真实 LangChain/OpenAI HTTP 序列化验证。"""
import json
import httpx
import pytest

from app.infrastructure.context_usage import context_usage_sink, evaluation_evidence_sink
from tests.native_model_helpers import client_model, messages, completion


def markers(payload):
    return [block["cache_control"] for message in payload["messages"]
            if isinstance(message.get("content"), list) for block in message["content"]
            if "cache_control" in block]


def stream_reply():
    chunks = [
        {"id": "s", "object": "chat.completion.chunk", "created": 1, "model": "fixture",
         "choices": [{"index": 0, "delta": {"content": "完成"}, "finish_reason": None}]},
        {"id": "s", "object": "chat.completion.chunk", "created": 1, "model": "fixture",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "s", "object": "chat.completion.chunk", "created": 1, "model": "fixture",
         "choices": [], "usage": completion()["usage"]},
    ]
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
        content="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n")


@pytest.mark.parametrize("streamed", [False, True])
async def test_explicit_cache_rejection_retries_once_and_meters_both_requests(tmp_path, streamed):
    requests, samples = [], []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(400, json={"error": {"message": "unsupported cache_control",
                                                       "param": "cache_control", "type": "invalid_request_error"}})
        return stream_reply() if streamed else httpx.Response(200, json=completion())
    model = await client_model(tmp_path, handler, stream=streamed)
    model.cache_mode = "explicit"
    source = messages()
    before = [message.model_dump() for message in source]
    token = context_usage_sink.set(samples.append)
    try:
        if streamed:
            chunks = [chunk async for chunk in model.astream(source)]
            assert "".join(chunk.text for chunk in chunks) == "完成"
        else:
            assert (await model.ainvoke(source)).content == "OK"
        assert len(requests) == 2 and markers(requests[0]) and not markers(requests[1])
        assert len(samples) == 2 and samples[0]["input_tokens"] is None
        assert samples[1]["input_tokens"] == 1500
        assert samples[1]["prompt_cache"]["cache_read_tokens"] == 900
        assert samples[1]["prompt_cache"]["cache_write_tokens"] == 512
        assert samples[1]["prompt_cache"]["cache_retry_without_markers"]
        assert [message.model_dump() for message in source] == before
        assert not model.gateway._semaphore.locked()
    finally:
        context_usage_sink.reset(token)
        await model.aclose()


async def test_cache_usage_and_opt_in_request_evidence_are_separate(tmp_path):
    requests, samples, evidence = [], [], []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion())
    model = await client_model(tmp_path, handler)
    model.cache_mode = "explicit"
    usage_token = context_usage_sink.set(samples.append)
    token = evaluation_evidence_sink.set(evidence.append)
    try:
        await model.ainvoke(messages())
        assert [item["kind"] for item in evidence] == ["model_request", "model_response_fragment", "model_response_contract"]
        assert evidence[0]["payload"]["messages"] == requests[0]["messages"]
        assert samples[0]["prompt_cache"]["cache_read_tokens"] == 900
        assert samples[0]["prompt_cache"]["cache_write_tokens"] == 512
        assert "核实商品" not in json.dumps(samples, ensure_ascii=False)
        assert "test-key" not in json.dumps(evidence)
    finally:
        evaluation_evidence_sink.reset(token)
        context_usage_sink.reset(usage_token)
        await model.aclose()


async def test_authentication_failure_is_not_cache_fallback(tmp_path):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "invalid credentials"}})
    model = await client_model(tmp_path, handler)
    model.cache_mode = "explicit"
    try:
        with pytest.raises(Exception):
            await model.ainvoke(messages())
        assert len(requests) == 1
    finally:
        await model.aclose()
