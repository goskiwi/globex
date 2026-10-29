"""实际子图与AG-UI适配器：来源校验拒绝的提交不能在过程里保留已完成。"""
import json
from ag_ui.core import RunAgentInput
from tests.test_agent_handoff import factories,call,submission,task,context
from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.infrastructure.eventbus import observe_run_events
from tests.native_tool_helpers import observe_tool_facts
from dataclasses import asdict

async def test_rejected_submission_is_not_reported_completed(tmp_path,monkeypatch,context):
 invalid=submission(candidates=[{'product_id': 'invented', 'evidence_fields': ['description', 'weight_kg']}])
 valid=submission(candidates=[{'product_id': 'P1001', 'evidence_fields': ['description', 'weight_kg']}])
 _,model,dispatch,bus=factories(tmp_path,monkeypatch,[call('read_candidates',{}),invalid,valid])
 body=RunAgentInput.model_validate({'threadId':'handoff','runId':'review','state':{},'messages':[{'id':'u','role':'user','content':'合成研究请求'}],'tools':[],'context':[],'forwardedProps':{}})
 adapter=AGUIRunAdapter(body,lambda event:None);adapter.start()
 events=[]
 def observe(event):
  events.append(event)
  adapter.on_trade_event(event)
 native=[]
 def native_event(event):
  native.append(event);adapter.on_agent_event(event)
 with observe_tool_facts(native_event), observe_run_events(observe):output=(await dispatch('search_agent',task())).data
 starts=[asdict(e) for e in native if e.type=='TOOL_CALL_START' and e.tool_call_name=='SubagentResult']
 assert output['status']=='completed' and model.cursor==3
 bad=starts[0]['tool_call_id'];step=next(s for s in adapter.state['process']['steps'] if s['id']=='review:tool:'+bad)
 print(json.dumps({'final_status':output['status'],'rejected_payload':starts[0]['arguments'],'displayed_step':step,
  'later_feedback':[m.content for m in model.seen[-1] if m.name=='handoff_feedback'],
  'raw_end_events':[asdict(e) for e in native if e.tool_call_id==bad and e.type=='TOOL_RESULT_END']},ensure_ascii=False))
 assert step['status']!='completed','来源校验拒绝的提交仍在执行过程显示已完成'
