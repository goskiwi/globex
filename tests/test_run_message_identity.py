"""审查消息主键边界：真实RunService/原生图/SQLite，脚本模型，无外部模型。"""
import json
from pathlib import Path
import pytest
from ag_ui.core import RunAgentInput
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.presentation.ag_ui import parse_intent
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_langgraph_runtime import _container

@pytest.mark.parametrize('reuse_id',[False,True])
async def test_new_run_with_reused_user_message_id(tmp_path,monkeypatch,reuse_id):
    model=ScriptedModel(responses=[
        call('set_shopping_requirements',{'changes':{'filters':{'price_max_major':300,'target_currency':'CNY'}}}),AIMessage(content='已处理第一轮。'),
        call('set_shopping_requirements',{'changes':{'filters':{'price_max_major':100}}}),AIMessage(content='已处理第二轮。')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        await c.runs.startup()
        for index,text in enumerate(['预算300元','预算改为100元']):
            body=RunAgentInput.model_validate({'threadId':'audit-session','runId':f'audit-run-{index}',
                'messages':[{'id':'fixed-user-id' if reuse_id else f'user-{index}','role':'user','content':text}],
                'state':{},'tools':[],'context':[],'forwardedProps':{'buyerId':'audit-buyer'}})
            assert parse_intent(body).raw_query==text
            await c.runs.start(body,'audit-buyer')
            await c.runs.wait(body.run_id,'audit-buyer',10)
        s=c.orchestrator._sessions._agents['audit-session'];state=(await s.graph.aget_state(s.config)).values
        user=[m for m in state['messages'] if isinstance(m,HumanMessage) and m.name=='audit-buyer']
        receipt=next(m for m in reversed(state['messages']) if isinstance(m,ToolMessage) and m.name=='set_shopping_requirements')
        data={'reuse_id':reuse_id,'model_calls':model.cursor,'saved_user_messages':[m.text for m in user],
            'latest_request':state['shopping_work']['latest_request'],'budget':state['shopping_work']['filters']['price_max_major'],
            'second_update_status':receipt.status,'second_update_result':receipt.content}
        assert len(user)==2 and data['budget']==100 and receipt.status=='success'
        assert data['latest_request']=='预算改为100元'
        saved=await c.runs.journal.session('audit-session','audit-buyer')
        ids=[m['id'] for m in saved['messages'] if m['role']=='user']
        assert len(ids)==2 and len(set(ids))==2 and 'fixed-user-id' not in ids
    finally:
        await c.shutdown()
