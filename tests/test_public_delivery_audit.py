"""公开组件、事件与最终文字使用同一审核，内部业务证据保持原样。"""
import json
import pytest
from langchain_core.messages import ToolMessage
from app.application.usecases.intents import SubmitIntentInput
from app.infrastructure.eventbus import observe_run_events
from app.infrastructure.eventbus import TradeEventBus,TradeEvent
from app.infrastructure.security.output_guard import audit_output
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container
from tests.test_product_delivery import adapter


def test_structured_output_audit_preserves_identifiers_amounts_and_input():
    marker='sk-'+'x'*24
    value={'guidance':marker,'hits':[{'product_id':'P1003','sku_id':'P1003-S1','price_major':129,'weight_kg':0.38}],
           'snapshot_hash':'a'*64,'reason':['说明 '+marker]}
    safe,public=audit_output(value)
    assert not safe and marker not in json.dumps(public)
    assert value['guidance']==marker and public['hits']==value['hits'] and public['snapshot_hash']==value['snapshot_hash']


def test_public_event_projection_does_not_replace_internal_evidence():
    bus=TradeEventBus();queue=bus.subscribe('s');marker='sk-'+'x'*24
    raw={'tool':'product_search_tool','hits':[{'product_id':'P1003','title':marker}]}
    public={**raw,'hits':audit_output(raw['hits'])[1]}
    bus.publish('s','tool.result',raw,public_payload=public)
    event=queue.get_nowait()
    assert event.payload==raw and event.to_dict()['payload']==public
    assert marker not in json.dumps(event.to_dict())
    assert TradeEvent.from_dict(event.to_dict()).payload==public


@pytest.mark.parametrize('enabled',[True,False])
async def test_component_and_lifecycle_cannot_bypass_configured_output_policy(tmp_path,monkeypatch,enabled):
    monkeypatch.setenv('OUTPUT_GUARD_ENABLED','1' if enabled else '0')
    marker='sk-'+'SYNTHETICONLY'*3
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),call('recommend_products',
        {'picks':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'依据目录选择'}],
         'guidance':'合成审查标记：'+marker},turn_action='finish')])
    c=await _container(tmp_path,monkeypatch,model)
    catalog=c.orchestrator._sessions._main_factory._search_factory._catalog_search
    catalog._embedder=catalog._vector_index=catalog._reranker=None
    target,frames=adapter()
    try:
        with observe_run_events(target.on_trade_event):
            result=await c.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','推荐一款背包，不下单',run_id='audit-output'))
        target.finish(result.final_text,result.status,result.stop_reason)
        public=json.dumps(result.presentations,ensure_ascii=False)+json.dumps(target.state,ensure_ascii=False)+''.join(f.model_dump_json() for f in frames)
        assert (marker not in public) is enabled
        assert (marker not in result.final_text) is enabled
        assert model.cursor==2 and result.error is None
        card=result.presentations[0]['payload']['hits'][0]
        assert card['default_sku_id']=='P1003-S1' and card['weight_kg']==0.38
        state=(await c.orchestrator._sessions._agents['s'].graph.aget_state(c.orchestrator._sessions._agents['s'].config)).values
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='recommend_products')
        assert marker in receipt.artifact['data']['guidance']
        stored=await c.orchestrator._evidence_store.search('b','s',kind='recommendation')
        assert marker in stored[0]['data']['guidance']
    finally:await c.shutdown()
