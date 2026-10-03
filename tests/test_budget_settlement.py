"""唯一模型调用边界的原生 usage 结算与预算分档。"""
import time
import asyncio
import json
import httpx
import pytest
from app.infrastructure.budget import init_budget, MINIMAL_MODE_HINT
from app.infrastructure.langchain_model import ChatModelAdapter
from app.infrastructure.operational_metrics import begin_request, finish_request, observe_model_started
from tests.native_model_helpers import client_model, completion
from app.infrastructure.throttle import GatewayThrottle


@pytest.mark.parametrize('usage,expected',[
    ({'input_tokens':10},None),({'output_tokens':10},None),
    ({'input_tokens':-1,'output_tokens':1},None),
    ({'input_tokens':True,'output_tokens':1},None),
    ({'input_tokens':'10','output_tokens':1},None),
    ({'input_tokens':float('inf'),'output_tokens':1},None),
    ({'input_tokens':float('nan'),'output_tokens':1},None),
    ({'input_tokens':0,'output_tokens':0},0),
    ({'input_tokens':10,'output_tokens':5},15),
    ({'total_tokens':20},None),
])
async def test_native_settlement_requires_complete_integer_evidence(usage,expected):
    budget=init_budget(10000)
    observation=begin_request()
    try:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        raw=completion()
        raw['usage']={'prompt_tokens':usage.get('input_tokens'),'completion_tokens':usage.get('output_tokens')}
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(model_dump=lambda:raw)))))
        model=ChatModelAdapter(client=client, model_name='fixture', streaming=False, max_tokens=128)
        await model.ainvoke('查询')
        assert budget.used==expected if expected is not None else budget.used>128
        assert budget.reserved==0
        summary=finish_request(observation)
        assert summary['usage_complete']==(expected is not None)
        assert summary['model_calls']==1
        if expected is None: assert summary['input_tokens'] is None
    finally:
        init_budget(0)
        if not observation.finished:finish_request(observation,'failed')


def test_late_settlement_does_not_rewrite_finished_request_metrics():
    observation=begin_request()
    observe_model_started()
    summary=finish_request(observation,'cancelled')
    from app.infrastructure.operational_metrics import observe_model
    observe_model(input_tokens=10,output_tokens=2,elapsed_ms=1,ttft_ms=None,cost_usd=None)
    assert summary['usage_complete'] is False and summary['input_tokens'] is None


@pytest.mark.parametrize('used,expected_model,minimal',[(0,'main',False),(12000,'lite',False),(17000,'lite',True),(19500,None,False)])
async def test_budget_routes_before_reservation_and_preserves_model_instance(tmp_path,used,expected_model,minimal):
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    model.model_name='main'
    budget=init_budget(20000)
    budget.charge('prior',used)
    try:
        if expected_model is None:
            from app.application.runtime.errors import ExecutionStopped
            with pytest.raises(ExecutionStopped):
                await model.ainvoke('查询')
        else:
            from langchain.agents import create_agent
            from app.application.runtime.execution import ExecutionMiddleware
            graph=create_agent(model,middleware=[ExecutionMiddleware(4,budget_lite_model='lite')])
            result=await graph.ainvoke({'messages':[{'role':'user','content':'查询'}]})
        assert model.model_name=='main' and budget.reserved==0
        if expected_model is None:
            assert not requests
        else:
            assert requests[0]['model']==expected_model
            assert any(MINIMAL_MODE_HINT in (m.get('content') or '') for m in requests[0]['messages']) is minimal
            assert budget.used==used+1510
            assert requests[0]['max_completion_tokens'] < 20000-used
    finally:
        init_budget(0)
        await model.aclose()


@pytest.mark.parametrize('cancelled',[False,True])
async def test_settlement_observation_error_releases_slot_and_preserves_cancellation(tmp_path,monkeypatch,cancelled):
    entered=asyncio.Event()
    async def handler(request):
        if cancelled:
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    throttle=GatewayThrottle(1,0)
    model.gateway=throttle
    def broken(*args,**kwargs):
        raise OverflowError('模拟计量异常')
    monkeypatch.setattr('app.infrastructure.langchain_model.record_context_usage',broken)
    budget=init_budget(10000)
    task=asyncio.create_task(model.ainvoke('查询'))
    try:
        if cancelled:
            await asyncio.wait_for(entered.wait(),2)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancelled else OverflowError):
            await task
        assert budget.reserved==0
        async with asyncio.timeout(1), throttle.slot():pass
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        init_budget(0)
        await model.aclose()


@pytest.mark.parametrize('stream',[False,True])
async def test_adapter_does_not_reroute_structured_calls_after_context_assembly(tmp_path,stream):
    from tests.test_structured_model_usage import Summary, response
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return response(stream)
    model=await client_model(tmp_path,handler,stream=stream)
    model.model_name='main'
    budget=init_budget(20000);budget.charge('prior',12000)
    try:
        # 结构化摘要不自行选择降档；调用方明确传入模型，适配器不暗改请求。
        structured=model.with_structured_output(Summary,method='function_calling')
        assert (await structured.ainvoke('查询')).goal=='旅行'
        assert requests[0]['model']=='main'
        init_budget(20000)
        assert (await structured.ainvoke('查询')).goal=='旅行'
        assert requests[1]['model']=='main'
        assert requests[0]['tools']==requests[1]['tools']
        assert requests[0]['tool_choice']==requests[1]['tool_choice']
    finally:
        init_budget(0)
        await model.aclose()
