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
            result,events,response=await submit_stream(client,'http://test','stream-session','buyer','推荐背包，不下单')
            validate_smoke_result(result,events)
            assert any(e['type']=='usage.summary' for e in events)
            run_id=response.headers['x-run-id']
            replay=await client.get(f'/commerce/ag-ui/runs/{run_id}/events',params={'buyer_id':'buyer'})
            restored,trace,_=read_stream(replay.text)
            assert restored==result and trace==events and model.cursor==2
            saved=await client.get(f'/commerce/ag-ui/runs/{run_id}',params={'buyer_id':'buyer'})
            assert 'input' not in saved.json()
    finally:await c.shutdown()

@pytest.mark.parametrize('frames',[
    [{'type':'RUN_STARTED'}],
    [{'type':'RUN_STARTED'},{'type':'RUN_ERROR','message':'服务中断'}],
])
def test_script_never_calls_incomplete_or_failed_stream_success(frames):
    with pytest.raises(ValueError):read_stream(''.join('data: '+json.dumps(f)+'\n\n' for f in frames))
