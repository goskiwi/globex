"""脚本使用实际持久运行/SSE，终态、业务证据与重放一致。"""
import json
import pytest
import httpx
from fastapi import FastAPI
from app.presentation.ag_ui import register_ag_ui_routes
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container
from scripts.run_stream import submit_stream,read_stream
from scripts.smoke_e2e import validate_smoke_result

async def test_scripts_consume_persisted_public_run_without_side_subscription(tmp_path,monkeypatch):
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),call('recommend_products',{
        'picks':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'依据已读资料'}],
        'guidance':'查看这款背包。'},turn_action='finish')])
    c=await _container(tmp_path,monkeypatch,model)
    catalog=c.orchestrator._sessions._main_factory._search_factory._catalog_search
    catalog._embedder=catalog._vector_index=catalog._reranker=None
    app=FastAPI();register_ag_ui_routes(app,lambda:c.orchestrator,get_runtime=lambda:c.runs)
    try:
        await c.runs.startup()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            stream,response=await submit_stream(client,'http://test','stream-session','buyer','推荐背包，不下单')
            stream.require_success()
            result,events={"final_text":stream.final_text},stream.frames
            validate_smoke_result(result,events)
            assert [e['name'] for e in events if e['type']=='CUSTOM']==['usage.summary']
            assert sum(e['type']=='STEP_STARTED' for e in events)==2
            assert not any(e['type'] in {'tool.invoke','tool.result','presentation.result','final.result'} for e in events)
            run_id=response.headers['x-run-id']
            replay=await client.get(f'/commerce/ag-ui/runs/{run_id}/events',params={'buyer_id':'buyer'})
            restored=read_stream(replay.text)
            assert restored.final_text==stream.final_text and restored.frames==events and model.cursor==2
            saved=await client.get(f'/commerce/ag-ui/runs/{run_id}',params={'buyer_id':'buyer'})
            assert 'input' not in saved.json()
    finally:await c.shutdown()

@pytest.mark.parametrize('frames',[
    [{'type':'RUN_STARTED'}],
    [{'type':'RUN_STARTED'},{'type':'RUN_ERROR','message':'服务中断'}],
])
def test_script_never_calls_incomplete_or_failed_stream_success(frames):
    stream=read_stream(''.join('data: '+json.dumps(f)+'\n\n' for f in frames))
    assert stream.frames==frames
    with pytest.raises(ValueError):stream.require_success()

@pytest.mark.parametrize('ending',['RUN_ERROR','json_fragment','disconnect'])
async def test_failed_case_keeps_receipt_before_terminal_or_transport_failure(ending):
    from tests.native_trace_helpers import tool_frames
    from scripts.eval.trace import tool_observations
    from scripts import eval_regression
    from app.infrastructure.identity import IdentityPolicy
    frames=[{'type':'RUN_STARTED','threadId':'thread','runId':'run'},
        *tool_frames('create_order_tool',{'confirmation_required':True,'confirmation':{'confirmation_id':'synthetic','status':'pending'}})]
    wire=''.join('data: '+json.dumps(f)+'\n\n' for f in frames)
    if ending=='RUN_ERROR':wire+='data: '+json.dumps({'type':'RUN_ERROR','message':'合成后续故障','code':'AGENT_ERROR'})+'\n\n'
    elif ending=='json_fragment':wire+='data: {"type":'
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield wire.encode()
            if ending=='disconnect':raise httpx.ReadError('synthetic transport failure')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=Body()))) as client:
        with pytest.raises(eval_regression.CaseExecutionError) as error:
            await eval_regression.run_case(client,{'id':'failure','description':'故障保留证据','queries':['合成请求'],
                'rubric':{'p0':[],'p1':[],'p2':[]}},'',identity_policy=IdentityPolicy(mode='hmac',secret='synthetic-local-identity-for-testing'))
    receipt=tool_observations(error.value.trace_events)[0]
    assert receipt['name']=='create_order_tool' and receipt['data']['confirmation']['confirmation_id']=='synthetic'
    assert '合成请求' in error.value.transcript


def test_snapshot_publication_skips_unchanged_state_without_aliasing():
    from app.application.agents.ag_ui_adapter import AGUIRunAdapter
    from ag_ui.core import RunAgentInput
    events=[]
    adapter=AGUIRunAdapter(RunAgentInput.model_validate({'threadId':'s','runId':'r','state':{},'tools':[],
        'context':[],'messages':[{'id':'u','role':'user','content':'合成请求'}],'forwardedProps':{'buyerId':'b'}}),events.append)
    adapter.snapshot();adapter.snapshot()
    assert len(events)==1
    adapter.state['status']='running';adapter.snapshot()
    assert len(events)==2 and events[0].snapshot['status']=='queued'
