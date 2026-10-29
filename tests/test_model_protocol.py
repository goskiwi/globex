"""真实HTTP序列化边界：违约调用不交给Agent执行，消耗和响应模型仍完整计量。"""
import json
import httpx
import pytest
from app.infrastructure.model_protocol import ModelProtocolViolation
from app.infrastructure.context_usage import context_usage_sink
from tests.native_model_helpers import client_model, completion, messages

TOOLS=[{'type':'function','function':{'name':n,'description':'test','parameters':{'type':'object','properties':{}}}} for n in ('lookup','order')]

def called(*names):
    raw=completion()
    raw['model']='served-model'
    raw['choices'][0]['message']={'role':'assistant','content':None,'tool_calls':[
        {'id':str(i),'type':'function','function':{'name':n,'arguments':'{}'}} for i,n in enumerate(names)]}
    raw['choices'][0]['finish_reason']='tool_calls'
    return raw

@pytest.mark.parametrize('tools,choice,names,code',[
    (TOOLS,'none',['lookup'],'tools_forbidden'),
    ([],None,['lookup'],'tools_forbidden'),
    (TOOLS,None,['lookup','unknown'],'undeclared_tool'),
    (TOOLS,{'type':'function','function':{'name':'lookup'}},['order'],'wrong_forced_tool'),
])
async def test_illegal_completions_rejected_and_usage_settled(tmp_path,tools,choice,names,code):
    model=await client_model(tmp_path,lambda r:httpx.Response(200,json=called(*names)))
    samples=[];token=context_usage_sink.set(samples.append)
    try:
        with pytest.raises(ModelProtocolViolation) as error:
            await model.ainvoke(messages(),tools=tools,tool_choice=choice)
        assert error.value.code==code
        assert len(samples)==1 and samples[0]['input_tokens']==1500
        assert samples[0]['prompt_cache']['response_model']=='served-model'
        assert samples[0]['prompt_cache']['response_model_matches'] is False
        assert samples[0]['prompt_cache']['protocol_status']==code
        assert not model.gateway._semaphore.locked()
    finally:
        context_usage_sink.reset(token);await model.aclose()

async def test_valid_parallel_tools_are_preserved(tmp_path):
    model=await client_model(tmp_path,lambda r:httpx.Response(200,json=called('lookup','order')))
    try:
        result=await model.ainvoke(messages(),tools=TOOLS)
        assert [call["name"] for call in result.tool_calls]==['lookup','order']
    finally:await model.aclose()

@pytest.mark.parametrize('illegal', [False,True,'incomplete'])
async def test_streamed_tool_batch_is_atomic_and_usage_known(tmp_path,illegal):
    raw=called('lookup')
    chunks=[{'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[
        {'index':0,'delta':{'tool_calls':[{'index':0,'id':'a','type':'function','function':{'name':'look','arguments':'{'}}]},'finish_reason':None}]},
        {'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[
        {'index':0,'delta':{'tool_calls':[{'index':0,'function':{'name':'up','arguments':'}'}}]},'finish_reason':None}]}]
    if illegal is True:
        chunks.append({'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[
            {'index':0,'delta':{'tool_calls':[{'index':1,'id':'bad','type':'function','function':{'name':'unknown','arguments':'{}'}}]},'finish_reason':None}]})
    if illegal != 'incomplete':
        chunks.append({'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})
    chunks.append({'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[],'usage':raw['usage']})
    response=lambda r:httpx.Response(200,headers={'content-type':'text/event-stream'},content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    model=await client_model(tmp_path,response,stream=True)
    emitted=[];samples=[];token=context_usage_sink.set(samples.append)
    try:
        stream=model.astream(messages(),tools=TOOLS)
        async def consume():
            async for part in stream:emitted.append(part)
        if illegal:
            with pytest.raises(ModelProtocolViolation):await consume()
            assert not any(part.tool_call_chunks for part in emitted)
        else:
            await consume()
            combined = emitted[0]
            for part in emitted[1:]:
                combined = combined + part
            assert combined.tool_calls[0]["name"] == "lookup"
        assert len(samples)==1 and samples[0]['input_tokens']==1500
        assert not model.gateway._semaphore.locked()
    finally:context_usage_sink.reset(token);await model.aclose()

async def test_truncated_nonstream_tool_is_not_executable(tmp_path):
    raw=called('order');raw['choices'][0]['finish_reason']='length'
    model=await client_model(tmp_path,lambda r:httpx.Response(200,json=raw))
    try:
        with pytest.raises(ModelProtocolViolation) as error:
            await model.ainvoke(messages(),tools=TOOLS)
        assert error.value.code=='incomplete_tool_response'
    finally:await model.aclose()


async def test_required_product_recheck_cannot_be_replaced_by_plain_answer(tmp_path):
    tools = [{'type': 'function', 'function': {'name': 'product_search_tool',
              'parameters': {'type': 'object', 'properties': {}}}}]
    model = await client_model(tmp_path, lambda r: httpx.Response(200, json=completion()), stream=False)
    try:
        with pytest.raises(ModelProtocolViolation) as error:
            await model.ainvoke(messages(), tools=tools,
                tool_choice={"type": "function", "function": {"name": "product_search_tool"}})
        assert error.value.code == 'required_tool_missing'
        assert model.streaming is False
    finally:
        await model.aclose()

@pytest.mark.parametrize('stream',[False,True])
async def test_duplicate_ids_rejected_before_business_graph_and_followups_converge(tmp_path,monkeypatch,stream):
    from tests.test_langgraph_runtime import _container
    from tests.test_execution_facts import request
    from app.application.runtime.tool_history import tool_call_records
    from scripts.eval.trace import tool_observations
    monkeypatch.setenv('PROMPT_PIN_VERSION','')
    raw=called('product_search_tool','product_search_tool')
    for call,sku in zip(raw['choices'][0]['message']['tool_calls'],('P1003-S1','P1001-S1')):
        call['id']='duplicate';call['function']['arguments']=json.dumps({'sku_id':sku})
    requests=[]
    def response(req):
        requests.append(json.loads(req.content))
        body=raw if len(requests)==1 else completion()
        if not stream:return httpx.Response(200,json=body)
        message=body['choices'][0]['message']
        delta={**message}
        if 'tool_calls' in delta:
            delta['tool_calls']=[{'index':i,**call} for i,call in enumerate(delta['tool_calls'])]
        chunks=[{'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[
            {'index':0,'delta':delta,'finish_reason':body['choices'][0]['finish_reason']}]},
            {'id':'s','object':'chat.completion.chunk','created':1,'model':'served-model','choices':[],'usage':body['usage']}]
        return httpx.Response(200,headers={'content-type':'text/event-stream'},content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    model=await client_model(tmp_path,response,stream=stream)
    c=await _container(tmp_path,monkeypatch,model)
    executions=[]
    from app.application.usecases.catalog_search import CatalogSearchUseCase
    original=CatalogSearchUseCase.execute
    async def measured(self,*args,**kwargs):
        executions.append(True)
        return await original(self,*args,**kwargs)
    monkeypatch.setattr(CatalogSearchUseCase,'execute',measured)
    try:
        await c.runs.startup()
        for index in range(3):
            run_id=f'duplicate-{index}'
            await c.runs.start(request(run_id),'buyer')
            run=await c.runs.wait(run_id,'buyer',10)
            assert run['status']==('error' if index==0 else 'completed')
            rows,_,_=await c.runs.journal.events(run_id,'buyer')
            assert not tool_observations([r['event'] for r in rows])
        assert not executions and len(requests)==3
        state=await c.orchestrator._sessions._agents['facts-session'].graph.aget_state(
            c.orchestrator._sessions._agents['facts-session'].config)
        assert not state.next and not tool_call_records(state.values['messages'])
    finally:
        await c.shutdown();await model.aclose()
