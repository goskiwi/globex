"""原生图覆盖提前拒绝、非法调用、正常工具与跨轮身份，不依赖私有采集器。"""
import json
import pytest
from ag_ui.core import RunAgentInput
from langchain_core.messages import AIMessage
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container
from scripts.eval.trace import tool_observations,model_starts


def request(run):
    return RunAgentInput.model_validate({'threadId':'facts-session','runId':run,'state':{},'tools':[], 'context':[],
        'messages':[{'id':'u','role':'user','content':'合成请求，不下单'}],'forwardedProps':{'buyerId':'buyer'}})


@pytest.mark.parametrize('kind',['invalid_json','mixed_invalid','mixed_memory','unknown_tool','typed_error','state_conflict'])
async def test_all_rejection_origins_share_native_receipts(tmp_path,monkeypatch,kind):
    invalid={'id':'bad','name':'product_search_tool','args':'{"normalized_query":','error':'synthetic'}
    first={
        'invalid_json':AIMessage(content='',invalid_tool_calls=[invalid]),
        'mixed_invalid':AIMessage(content='',tool_calls=[{'id':'valid','name':'product_search_tool','args':{'product_id':'P1003'}}],invalid_tool_calls=[invalid]),
        'mixed_memory':AIMessage(content='',tool_calls=[{'id':'memory','name':'remember_preference_tool','args':{'kind':'like','statement':'喜欢蓝色'}},
            {'id':'search','name':'product_search_tool','args':{'product_id':'P1003'}}]),
        'unknown_tool':call('invented_tool',{}),
        'typed_error':call('product_search_tool',{'top_k':0}),
        'state_conflict':AIMessage(content='',tool_calls=[{'id':'change','name':'set_shopping_requirements','args':{'changes':{'filters':{'price_max_major':100}}}},
            {'id':'search','name':'product_search_tool','args':{'product_id':'P1003'}}]),
    }[kind]
    model=ScriptedModel(responses=[first,AIMessage(content='合成错误已说明。')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        await c.runs.startup()
        await c.runs.start(request(kind),'buyer')
        run=await c.runs.wait(kind,'buyer',10)
        rows,_,_=await c.runs.journal.events(kind,'buyer')
        frames=[r['event'] for r in rows];receipts=tool_observations(frames)
        assert run['status']=='completed' and model.cursor==2 and len(model_starts(frames))==2
        assert len(receipts)==len(first.tool_calls)+len(first.invalid_tool_calls)
        assert all(r['status']=='error' for r in receipts)
        if kind!='unknown_tool':assert all(r['failure'].get('executed') is False for r in receipts)
        if kind in {'invalid_json','mixed_invalid'}:
            assert next(r for r in receipts if r['name']=='product_search_tool' and isinstance(r['arguments'],str))['arguments']==invalid['args']
        assert len(run['state']['process']['steps'])==len(receipts)
        assert not any(f.get('name')=='tool.lifecycle' for f in frames)
        replay,_,_=await c.runs.journal.events(kind,'buyer')
        assert replay==rows and model.cursor==2
    finally:await c.shutdown()

async def test_nested_graph_rejection_is_recorded_before_subtask_failure(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[AIMessage(content='',invalid_tool_calls=[{
        'id':'child-bad','name':'product_search_tool','args':'{"normalized_query":','error':'synthetic'}])])
    parent=ScriptedModel(responses=[call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'合成独立研究'}}),
        AIMessage(content='子任务未完成，已说明缺口。')])
    c=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**k:child)
    try:
        await c.runs.startup()
        await c.runs.start(request('nested-failure'),'buyer')
        run=await c.runs.wait('nested-failure','buyer',10)
        rows,_,_=await c.runs.journal.events('nested-failure','buyer')
        receipts=tool_observations([r['event'] for r in rows])
        child_receipt=next(r for r in receipts if r['name']=='product_search_tool')
        assert child_receipt['failure']=={'code':'invalid_arguments','executed':False}
        assert child_receipt['index']<next(r['index'] for r in receipts if r['name']=='task_dispatch')
        assert len({r['call_id'] for r in receipts})==len(receipts)==2
        assert child.cursor==1 and parent.cursor==2
    finally:await c.shutdown()
