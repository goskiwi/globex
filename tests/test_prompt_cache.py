"""缓存验收在真实 SDK/HTTP 序列化边界断言，不用拼出的假请求代替实际调用。"""
from copy import deepcopy
from dataclasses import replace
import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, convert_to_messages
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.prompt_cache import mark_messages, read_cache_usage
from app.infrastructure.context_usage import context_usage_sink, context_call_kind
from app.infrastructure.budget import TokenBudget, _budget_var
from app.infrastructure.throttle import GatewayThrottle
from tests.test_retrieval import _settings


def settings(tmp_path, **kwargs):
    return replace(_settings(tmp_path), llm_api_key='test-cache-key', llm_base_url='https://cache.test/v1', llm_fallback_model='', llm_max_retries=0,
                   prompt_cache_mode='explicit', prompt_cache_policy='static_history', **kwargs)


def messages():
    return convert_to_messages([
        ('system','固定规则：必须核实 SKU。'),('user','搜索背包'),
        ('assistant','候选 P1，规格 A。'),('user','请再核价')])


def marks(body):
    return [(i,j) for i,m in enumerate(body['messages']) if isinstance(m.get('content'),list)
            for j,b in enumerate(m['content']) if 'cache_control' in b]


def completion(structured=False, usage=True):
    msg={'role':'assistant','content':'OK'}
    if structured:msg={'role':'assistant','content':None,'tool_calls':[{'id':'summary','type':'function',
        'function':{'name':'Summary','arguments':'{"goal":"背包"}'}}]}
    return {'id':'result','object':'chat.completion','created':1,'model':'fixture','choices':[
        {'index':0,'message':msg,'finish_reason':'stop'}], 'usage':{
        'prompt_tokens':1500,'completion_tokens':10,'total_tokens':1510,
        'prompt_tokens_details':{'cached_tokens':900,'cache_creation_input_tokens':512},
        'model_cost':0.001} if usage else None}


def streaming(raw):
    chunks=[{'id':'s','object':'chat.completion.chunk','created':1,'model':'fixture','choices':[
        {'index':0,'delta':{'content':'OK'},'finish_reason':'stop'}]},
        {'id':'s','object':'chat.completion.chunk','created':1,'model':'fixture','choices':[], 'usage':raw['usage']}]
    return httpx.Response(200,headers={'content-type':'text/event-stream'},
                          content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')


async def client_model(tmp_path, handler, stream=False, **kwargs):
    client=AsyncOpenAI(api_key="test-cache-key",base_url="https://cache.test/v1",
        max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return create_chat_model(settings(tmp_path,**kwargs),stream=stream,throttle=GatewayThrottle(1,0),client=client)


@pytest.mark.asyncio
async def test_synthetic_evidence_captures_final_request_only_when_explicitly_scoped(tmp_path):
    from app.infrastructure.context_usage import evaluation_evidence_sink
    requests, evidence, usage = [], [], []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion())
    model = await client_model(tmp_path, handler)
    sink = context_usage_sink.set(usage.append)
    try:
        await model.ainvoke(messages())
        token = evaluation_evidence_sink.set(evidence.append)
        try:
            await model.ainvoke(messages())
        finally:
            evaluation_evidence_sink.reset(token)
        await model.ainvoke(messages())
        assert [e['kind'] for e in evidence] == ['model_request', 'model_response_fragment', 'model_response_contract']
        body = evidence[0]['payload']
        assert body['messages'] == requests[1]['messages'] and marks(body)
        assert set(body) <= {'model','messages','tools','tool_choice','response_format','temperature','max_tokens','max_completion_tokens','stream','stream_options'}
        assert 'test-cache-key' not in json.dumps(evidence)
        assert '固定规则' not in json.dumps(usage, ensure_ascii=False)
    finally:
        context_usage_sink.reset(sink)
        await model.aclose()


@pytest.mark.asyncio
async def test_formatter_round_trip_preserves_state_and_tool_pairs(tmp_path):
    source=messages()[:-1]+[AIMessage(content="",tool_calls=[
        {"id":"a","name":"lookup","args":{"sku":"A"}},{"id":"b","name":"lookup","args":{"sku":"B"}}]),
        ToolMessage(tool_call_id="a",content="结果 A"),
        ToolMessage(tool_call_id="b",content="结果 B")]
    before=deepcopy([m.model_dump() for m in source])
    model=create_chat_model(settings(tmp_path),stream=False, client=create_chat_client(settings(tmp_path)))
    model.cache_mode="passthrough"
    plain=model.prepare_request(source)["messages"]
    model.cache_mode="explicit"
    marked=model.prepare_request(source)["messages"]
    assert len(marks({'messages':marked}))==3
    assert marks({'messages':marked})[-1][0]==len(marked)-1
    for m,original in zip(marked,plain):
        if isinstance(m.get('content'),list):
            for b in m['content']:b.pop('cache_control',None)
            if isinstance(original.get('content'),str):m['content']=m['content'][0]['text']
    assert marked==plain
    assert [m.model_dump() for m in source]==before
    model.cache_mode="passthrough"
    assert model.prepare_request(source)["messages"]==plain
    await model.aclose()


def test_no_boundary_inside_pending_or_malformed_parallel_tools():
    rows=[{'role':'system','content':'rules'}, {'role':'assistant','tool_calls':[{'id':'a'},{'id':'b'}]},
          {'role':'tool','tool_call_id':'a','content':'A'}]
    assert len(marks({'messages':mark_messages(rows,'static_history')[0]}))==1
    rows += [{'role':'user','content':'仍在审批'}, {'role':'assistant','content':'不应当作完成'}]
    assert len(marks({'messages':mark_messages(rows,'static_history')[0]}))==1


def test_actual_content_block_count_and_latest_user_not_marked():
    rows=[{'role':'system','content':[{'type':'text','text':str(i)} for i in range(25)]},
          {'role':'assistant','content':'一轮'}, {'role':'assistant','content':'二轮'},
          {'role':'user','content':'最新请求'}]
    marked,positions=mark_messages(rows,'static_history')
    assert positions==[24,25,26] and marks({'messages':marked})==[(0,24),(1,0),(2,0)]
    assert rows[-1]==marked[-1]
    # 摘要/裁剪后的新工作集重新计算，不保留旧位置或旧正文。
    assert mark_messages([rows[0],rows[-1]],'static_history')[1]==[24]


@pytest.mark.parametrize('data,read,write,invalid',[
    ({},None,None,False),
    ({'prompt_tokens_details':{'cached_tokens':0,'cache_creation_input_tokens':0}},0,0,False),
    ({'prompt_tokens_details':{'cached_tokens':7,'cache_write_tokens':8}},7,8,False),
    ({'prompt_tokens_details':{'cache_creation':{'ephemeral_5m_input_tokens':8}}},None,8,False),
    ({'prompt_tokens':10,'prompt_tokens_details':{'cached_tokens':8,'cache_creation_input_tokens':8}},None,None,True),
    ({'prompt_tokens_details':{'cache_write_tokens':1,'cache_creation_input_tokens':2}},None,None,True),
    ({'prompt_tokens_details':{'cached_tokens':True,'cache_creation_input_tokens':-1}},None,None,False),
])
def test_usage_unknown_zero_aliases_and_conflicts(data,read,write,invalid):
    parsed=read_cache_usage(data)
    assert (parsed['cache_read_tokens'],parsed['cache_write_tokens'],parsed['cache_usage_invalid'])==(read,write,invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('known',[False,True])
async def test_real_sdk_request_and_usage_streaming(tmp_path,stream,known):
    requests=[];samples=[];budget=TokenBudget(100000)
    def handler(request):
        requests.append(json.loads(request.content))
        raw=completion()
        if not known:raw['usage'].pop('prompt_tokens_details')
        return streaming(raw) if stream else httpx.Response(200,json=raw)
    model=await client_model(tmp_path,handler,stream)
    token=context_usage_sink.set(samples.append);bt=_budget_var.set(budget)
    try:
        if stream:
            async for result in model.astream(messages(),max_completion_tokens=128):pass
        else:
            result=await model.ainvoke(messages(),max_completion_tokens=128)
        assert marks(requests[0])==[(0,0),(2,0)]
        assert len(samples)==1 and samples[0]['input_tokens']==1500
        assert (samples[0]['ttft_ms'] is not None) is stream
        assert samples[0]['prompt_cache']['cache_read_tokens']==(900 if known else None)
        assert samples[0]['prompt_cache']['cache_write_tokens']==(512 if known else None)
        assert budget.used==1510 and budget.reserved==0  # 缓存命中不能减掉容量/Token 预算。
    finally:
        context_usage_sink.reset(token);_budget_var.reset(bt);await model.aclose()


class Summary(BaseModel):
    goal:str


@pytest.mark.asyncio
@pytest.mark.parametrize('structured',[False,True])
async def test_parameter_rejection_reacquires_slot_and_counts_both_attempts(tmp_path,structured):
    requests=[];samples=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        if len(requests)==1:return httpx.Response(400,json={'error':{'message':'cache_control is not supported','param':'messages'}})
        return httpx.Response(200,json=completion(structured))
    model=await client_model(tmp_path,handler)
    token=context_usage_sink.set(samples.append);kind=context_call_kind.set('summary' if structured else 'business')
    try:
        call=model.with_structured_output(Summary,method="function_calling").ainvoke(messages()) if structured else model.ainvoke(messages())
        await asyncio.wait_for(call,2)
        assert len(requests)==2 and marks(requests[0]) and not marks(requests[1])
        assert [s['input_tokens'] for s in samples]==[None,1500]
        assert samples[-1]['prompt_cache']['cache_retry_without_markers']
        assert not model.gateway._semaphore.locked()
        assert all(s['kind']==('summary' if structured else 'business') for s in samples)
    finally:context_usage_sink.reset(token);context_call_kind.reset(kind);await model.aclose()


@pytest.mark.asyncio
async def test_unrelated_error_not_retried(tmp_path):
    requests=[]
    def handler(request):
        requests.append(request);return httpx.Response(400,json={'error':{'message':'budget invalid'}})
    model=await client_model(tmp_path,handler)
    try:
        with pytest.raises(Exception):await model.ainvoke(messages())
        assert len(requests)==1
    finally:await model.aclose()


@pytest.mark.asyncio
async def test_passthrough_does_not_add_markers_and_invalid_settings_rejected(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValueError):create_chat_model(replace(settings(tmp_path),prompt_cache_policy='oops'), client=None)
    plain=create_chat_model(replace(settings(tmp_path),prompt_cache_mode='passthrough'),stream=False, client=create_chat_client(replace(settings(tmp_path),prompt_cache_mode='passthrough')))
    try:assert not marks(plain.prepare_request(messages()))
    finally:await plain.aclose()


def test_cache_trace_whitelist_preserves_unknown_and_redacts_text():
    from app.infrastructure.tracing import _sanitize_attributes
    attrs={'globex.prompt_cache.cache_read_tokens_known':False,'globex.prompt_cache.cache_marker_count':2,
           'globex.context.call_kind':'summary','globex.prompt_cache.buyer_text':'个人原文'}
    clean=_sanitize_attributes(attrs)
    assert clean=={k:v for k,v in attrs.items() if k!='globex.prompt_cache.buyer_text'}


@pytest.mark.asyncio
async def test_running_result_and_rejected_approval_do_not_get_confused(tmp_path):
    base=[messages()[0],AIMessage(content="",tool_calls=[{"id":"x","name":"trade","args":{}}])]
    model=create_chat_model(settings(tmp_path),stream=False, client=create_chat_client(settings(tmp_path)))
    try:
        assert marks(model.prepare_request(base))==[(0,0)]
        denied=ToolMessage(tool_call_id="x",content="拒绝，不执行",status="error")
        assert marks(model.prepare_request(base+[denied]))==[(0,0),(2,0)]
    finally:await model.aclose()


@pytest.mark.asyncio
async def test_fallback_cache_rejection_does_not_reissue_primary(tmp_path):
    requests=[];samples=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        if body['model']=='primary':return httpx.Response(429,json={'error':{'message':'rate limit','type':'rate_limit'}})
        if marks(body):return httpx.Response(400,json={'error':{'message':'unknown parameter cache_control'}})
        return httpx.Response(200,json=completion())
    config=replace(settings(tmp_path),llm_model='primary',llm_fallback_model='backup')
    model=create_chat_model(config,stream=False,throttle=GatewayThrottle(1,0), client=create_chat_client(config))
    from app.application.runtime.middleware import GatewayModelMiddleware
    from langchain.agents.middleware import ModelRequest
    middleware=GatewayModelMiddleware(config,model.gateway, client=model.client)
    for m in [model,middleware.fallback]:
        m.streaming=False
        await m.aclose()
        m.client=AsyncOpenAI(api_key="test",base_url=config.llm_base_url,max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async def invoke(request):
        return await request.model.ainvoke(request.messages)
    request=ModelRequest(model=model,messages=messages())
    token=context_usage_sink.set(samples.append)
    try:
        await asyncio.wait_for(middleware.awrap_model_call(request,invoke),2)
        assert [r['model'] for r in requests]==['primary','backup','backup']
        assert len(samples)==3 and [s['input_tokens'] for s in samples]==[None,None,1500]
        assert requests[-1]["model"]=='backup'
        # 新调用不会沿用上一调用的 cache_disabled，也不会串到别的买家。
        await middleware.awrap_model_call(request,invoke)
        assert [bool(marks(r)) for r in requests]==[True,True,False]*2
    finally:
        context_usage_sink.reset(token);await model.aclose();await middleware.fallback.aclose()


@pytest.mark.asyncio
async def test_invalid_structured_reply_does_not_probe_cache_each_strategy(tmp_path):
    requests=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        if marks(body):return httpx.Response(400,json={'error':{'message':'cache_control not supported'}})
        return httpx.Response(200,json=completion())  # OK 不符合摘要合同。
    model=await client_model(tmp_path,handler)
    try:
        with pytest.raises(ValueError):
            await asyncio.wait_for(model.with_structured_output(Summary,method="function_calling").ainvoke(messages()),3)
        assert len(requests)==2
        assert [bool(marks(r)) for r in requests]==[True,False]
    finally:await model.aclose()


@pytest.mark.asyncio
async def test_partial_stream_error_never_retries_or_replays_tools(tmp_path):
    requests=[];samples=[];closed=[]
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunk={'id':'partial','choices':[{'index':0,'delta':{'content':'开始'},'finish_reason':None}],
                   'usage':completion()['usage']}
            yield ('data: '+json.dumps(chunk)+'\n\n').encode()
            raise httpx.ReadError('stream interrupted')
        async def aclose(self):closed.append(True)
    def handler(request):
        requests.append(request);return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=BrokenStream())
    model=await client_model(tmp_path,handler,stream=True)
    token=context_usage_sink.set(samples.append)
    try:
        stream=model.astream(messages())
        with pytest.raises(httpx.ReadError):
            async for _ in stream:pass
        assert len(requests)==1 and closed and not model.gateway._semaphore.locked()
        assert len(samples)==1 and samples[0]['prompt_cache']['cache_write_tokens']==512
    finally:context_usage_sink.reset(token);await model.aclose()


@pytest.mark.asyncio
async def test_concurrent_stream_usage_is_request_local(tmp_path):
    class Stream(httpx.AsyncByteStream):
        def __init__(self,read):self.read=read
        async def __aiter__(self):
            await asyncio.sleep(.01)
            raw=completion();raw['usage']['prompt_tokens_details']['cached_tokens']=self.read
            yield streaming(raw).content
    def handler(request):
        body=json.loads(request.content)
        read=100 if '甲' in json.dumps(body,ensure_ascii=False) else 200
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Stream(read))
    model=await client_model(tmp_path,handler,stream=True)
    model.gateway=GatewayThrottle(2,0)
    async def call(text):
        samples=[];token=context_usage_sink.set(samples.append)
        try:
            async for response in model.astream([messages()[0],HumanMessage(content=text)]):pass
            return samples[-1]["prompt_cache"]["cache_read_tokens"]
        finally:context_usage_sink.reset(token)
    try:assert await asyncio.gather(call('甲'),call('乙'))==[100,200]
    finally:await model.aclose()


def test_frozen_replay_and_unknown_cost_not_zero():
    from scripts.eval.prompt_cache import fixture, aggregate, check_answer
    source,expected=fixture('preference_withdrawn','isolated',changed=True)
    assert expected['budget']==180 and expected['material_rule']=='none'
    assert not check_answer({'answer':{**expected,'sku_id':'P1003-S2'}},expected)['passed']
    row={'strategy':'C','case':'preference_withdrawn','passed':False,'calls':[
        {'kind':'business','usage':[{'input_tokens':None,'output_tokens':None}], 'elapsed_ms':10}]}
    summary=aggregate([row])
    assert summary['strategies']['C']['gateway_reported_cost']['total'] is None
    assert summary['paired_scenario_count']==0


def test_report_counts_unknown_and_bootstraps_scenarios_instead_of_repeats():
    from scripts.eval.report_prompt_cache import measured_total, paired
    from scripts.eval.prompt_cache import CASES
    assert measured_total([{'calls': [{'usage': []}]}], 'input_tokens') == {
        'total': None, 'observed_sum': None, 'known_attempts': 0, 'unknown_attempts': 1}
    rows = [{'case': case, 'strategy': strategy, 'repeat': repeat, 'passed': strategy == 'C'}
            for case in CASES for strategy in ('A', 'C') for repeat in range(3)]
    result = paired(rows, 'C', lambda r: float(r['passed']))
    assert result == {'scenario_count': 8, 'mean_difference': 1.0, 'bootstrap_95_ci': [1.0, 1.0]}
    assert paired(rows[:-1], 'C', lambda r: float(r['passed']))['scenario_count'] == 7


def test_incomplete_experiment_never_promotes_cache():
    from scripts.eval.report_prompt_cache import build_report
    result = build_report({'rows': []})
    assert result['layout_complete'] is False
    assert all(g['status'] == 'NOT_APPROVED' and not g['all_scenarios_passed']
               and g['production_cost_reduction'] is None for g in result['enablement'].values())
