"""模型韧性只经过原生模型与 middleware，隔离 HTTP 而非重造模型包装器。"""
import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage
from app.application.runtime.middleware import GatewayModelMiddleware
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.throttle import GatewayThrottle
from tests.native_model_helpers import client_model, completion
from tests.test_retrieval import _settings
from tests.test_model_stream_lifecycle import ControlledWireStream, wire_model


async def test_stream_holds_slot_until_cancelled_then_next_call_enters(tmp_path):
    throttle=GatewayThrottle(1,0)
    wire=ControlledWireStream()
    first=wire_model(throttle,wire)
    entered=asyncio.Event()
    def handler(request):
        entered.set()
        return httpx.Response(200,json=completion())
    second=await client_model(tmp_path,handler)
    second.gateway=throttle
    async def consume():
        return [part async for part in first.astream('查询')]
    task=asyncio.create_task(consume())
    waiting=None
    try:
        await asyncio.wait_for(wire.reading.wait(),2)
        waiting=asyncio.create_task(second.ainvoke('继续'))
        await asyncio.sleep(.05)
        assert not entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(waiting,2)
        assert wire.closed.is_set() and entered.is_set()
    finally:
        task.cancel()
        if waiting: waiting.cancel()
        await asyncio.gather(task,*([waiting] if waiting else []),return_exceptions=True)
        await first.aclose()
        await second.aclose()


@pytest.mark.parametrize('status,retries,fallback,expected_calls',[
    (429,2,False,2), (400,2,False,1), (429,2,True,4), (429,1,False,2),
])
async def test_retry_fallback_and_business_error_at_http_boundary(tmp_path,monkeypatch,status,retries,fallback,expected_calls):
    calls=[]
    def handler(request):
        body=json.loads(request.content)
        calls.append(body['model'])
        success=(fallback and body['model']=='backup') or (not fallback and retries==2 and status==429 and len(calls)==2)
        return httpx.Response(200,json=completion()) if success else httpx.Response(status,json={
            'error':{'message':'429 rate limit' if status==429 else 'model_not_found','type':'error'}})
    primary=await client_model(tmp_path,handler)
    primary.model_name='primary'
    backup=await client_model(tmp_path,handler)
    backup.model_name='backup'
    settings=replace(_settings(tmp_path),llm_model='primary',llm_fallback_model='',llm_max_retries=retries)
    bus=TradeEventBus(); queue=bus.subscribe('s')
    middleware=GatewayModelMiddleware(settings,GatewayThrottle(1,0),bus, client=None)
    if fallback:
        middleware.settings=replace(settings,llm_fallback_model='backup')
        middleware.fallback=backup
    monkeypatch.setattr('app.application.runtime.middleware.asyncio.sleep',AsyncMock())
    request=ModelRequest(model=primary,messages=[HumanMessage(content='查询')],tools=[],state={},runtime=None)
    async def invoke(request):
        return await request.model.ainvoke(request.messages)
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        if status==400 or (retries==1 and not fallback):
            with pytest.raises(APIStatusError):
                await middleware.awrap_model_call(request,invoke)
        else:
            assert (await middleware.awrap_model_call(request,invoke)).text=='OK'
        assert len(calls)==expected_calls
        if fallback:
            assert calls==['primary']*3+['backup']
            event=queue.get_nowait()
            assert event.type=='model.fallback' and event.payload=={'from':'primary','to':'backup'}
    finally:
        ShoppingContext.reset(token)
        await primary.aclose()
        await backup.aclose()
