"""计划拒绝及模型终态经过正式RunService，验证公开事实与重启恢复。"""
import pytest
from langchain_core.messages import AIMessage
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container
from tests.test_execution_facts import request
from scripts.eval.trace import tool_observations
from app.infrastructure.ag_ui_journal import AGUIJournal


@pytest.mark.parametrize('kind',['duplicate','dependent'])
async def test_plan_rejections_are_public_and_never_start_children(tmp_path,monkeypatch,kind):
    steps=[{'id':'first','goal':'先研究背包'},{'id':'second','goal':'根据前项研究规格','depends_on':['first']}]
    parallel=AIMessage(content='',tool_calls=[{'id':str(i),'name':'task_dispatch',
        'args':{'subagent_type':'search_agent','task':{'step_id':step}}}
        for i,step in enumerate(['first','first'] if kind=='duplicate' else ['first','second'])])
    model=ScriptedModel(responses=[call('set_research_plan',{'plan':steps}),parallel,AIMessage(content='计划尚未执行，相关任务需要分别提交。')])
    c=await _container(tmp_path,monkeypatch,model)
    def forbidden(*args,**kwargs):raise AssertionError('拒绝批次不应启动子图或模型')
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',forbidden)
    try:
        await c.runs.startup();await c.runs.start(request('plan-'+kind),'buyer')
        result=await c.runs.wait('plan-'+kind,'buyer',10)
        rows,_,_=await c.runs.journal.events('plan-'+kind,'buyer')
        receipts=tool_observations([r['event'] for r in rows])
        rejected=[r for r in receipts if r['name']=='task_dispatch']
        assert len(rejected)==2 and all(r['status']=='error' and r['failure']['executed'] is False for r in rejected)
        assert all(('重复' if kind=='duplicate' else '依赖') in r['failure']['reason'] for r in rejected)
        assert model.cursor==3
        assert len({r['call_id'] for r in receipts})==3
    finally:await c.shutdown()


@pytest.mark.parametrize('kind,status,code',[
    ('refusal','failed','provider_refusal'),('empty','failed','empty_delivery'),('length','partial','output_limit')])
async def test_model_terminal_outcomes_are_identical_after_restart(tmp_path,monkeypatch,kind,status,code):
    reply=AIMessage(content='已取得部分资料。' if kind=='length' else '',
        response_metadata={'finish_reason':{'refusal':'content_filter','length':'length','empty':'stop'}[kind]})
    model=ScriptedModel(responses=[reply])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        await c.runs.startup();await c.runs.start(request('model-'+kind),'buyer')
        run=await c.runs.wait('model-'+kind,'buyer',10)
        assert run['result']['status']==status and run['result']['stop_reason']==code and len(model.seen)==1
        path=c.runs.journal.path
        session=await AGUIJournal(path).session('facts-session','buyer')
        assert session['run']['result']==run['result']
        if status=='failed':assert session['processes'][0]['error']['code']==code
        else:assert 'error' not in session['processes'][0]
    finally:await c.shutdown()
