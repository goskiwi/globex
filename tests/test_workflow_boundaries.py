"""生产主图验证独立任务、并发展示与每次委派的实际输入；目录与持久化使用真实实现。"""
import json
import pytest
from langchain_core.messages import AIMessage,ToolMessage
from tests.test_agent_handoff import ScriptedModel,call,submission
from tests.test_langgraph_runtime import _container
from app.application.usecases.intents import SubmitIntentInput

def rec(guidance='依据已读资料选择',task_id=None):
    pick={'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'依据目录候选'}
    if task_id:pick['task_id']=task_id
    return call('recommend_products', {'picks': [pick], 'guidance': guidance}, turn_action='finish')


async def ask(c,text):
    result=await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY',text))
    state=(await c.orchestrator._sessions._agents['s'].graph.aget_state(c.orchestrator._sessions._agents['s'].config)).values
    return result,state


async def test_parallel_presentations_through_full_production_factory(tmp_path,monkeypatch):
    recommend=rec().tool_calls[0]
    compare={'name':'compare_products','id':'compare','args':{'entries':[{'product_id':pid,'sku_id':pid+'-S1','quantity':1,'reason':'已读候选'} for pid in ['P1003','P1049']], 'preferred_sku_id':None,'dimensions':[],'guidance':'两款按已读资料比较','turn_action':'continue'}}
    model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[{'name':'product_search_tool','id':pid,'args':{'product_id':pid}} for pid in ['P1003','P1049']]),AIMessage(content='',tool_calls=[recommend,compare]),AIMessage(content='结束')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        result,state=await ask(c,'推荐一款背包，并同时给出两款的比较。')
        print(json.dumps({'error':result.error,'status':result.status,'presentations':len(state.get('presentations',[]))},ensure_ascii=False))
        assert result.error is None and len(result.presentations)==2
    finally:await c.shutdown()


async def test_valid_catalog_identifier_is_not_a_business_rejection(tmp_path,monkeypatch):
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),rec('按你指定的P1003展示这个候选。'),AIMessage(content='未交付')])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        result,state=await ask(c,'请推荐P1003-S1，不下单。')
        receipts=[{'status':m.status,'text':m.text} for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='recommend_products']
        print(json.dumps({'receipts':receipts,'presentations':result.presentations},ensure_ascii=False))
        assert any(p['kind']=='recommendation' for p in result.presentations)
    finally:await c.shutdown()


@pytest.mark.parametrize('role',['trade_agent','search_agent'])
async def test_saved_plan_does_not_block_independent_delegation(tmp_path,monkeypatch,role):
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1008','sku_id':'P1008-S1'}),submission(candidates=[{'product_id':'P1008','sku_id':'P1008-S1','assessment':'独立水杯研究'}])])
    if role=='trade_agent':
        child=ScriptedModel(responses=[call('query_order_tool',{'order_id':'synthetic-missing-order'}),submission(status='failed',issues=['订单未找到'])])
    parent=ScriptedModel(responses=[call('set_research_plan',{'plan': [{'id':'bag','goal':'保留背包研究'}]}),AIMessage(content='计划已保存，暂不执行。'),call('task_dispatch',{'subagent_type':role,'task':{'goal':'独立查询一个订单' if role=='trade_agent' else '独立研究水杯'}}),AIMessage(content='已返回')])
    c=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    monkeypatch.setattr('app.application.agents.trade_agent.create_chat_model',lambda *a,**kw:child)
    try:
        await ask(c,'保存背包研究计划，暂不执行。')
        result,state=await ask(c,'保留原计划，先让专家处理一个独立任务。')
        data=next(m.artifact['data'] for m in reversed(state['messages']) if isinstance(m,ToolMessage) and m.name=='task_dispatch')
        print(json.dumps({'role':role,'child_calls':child.cursor,'status':data['status'],'issues':data['issues'],'feedback':data['feedback']},ensure_ascii=False))
        assert child.cursor>0,'旧计划阻止独立任务进入专家'
    finally:await c.shutdown()


async def test_each_delegation_uses_its_own_context(tmp_path,monkeypatch):
    children=[ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003','sku_id':sid}),submission(candidates=[{'product_id':'P1003','sku_id':sid,'assessment':'按本份任务背景选择'}])]) for sid in ['P1003-S1','P1003-S2']]
    instances=iter(children)
    def dispatch(fact):return call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究背包候选','known_facts':[fact]}})
    parent=ScriptedModel(responses=[dispatch('本份报告只研究石墨黑'),dispatch('本份报告只研究雾霾蓝'),AIMessage(content='分别交回候选')])
    c=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:next(instances))
    try:
        result,state=await ask(c,'请分别研究同款黑色和蓝色的候选。')
        results=[m.artifact['data'] for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch']
        skus=[[c['sku_id'] for c in r['candidates']] for r in results]
        print(json.dumps({'child_calls':[c.cursor for c in children],'skus':skus},ensure_ascii=False))
        assert skus==[['P1003-S1'],['P1003-S2']]
    finally:await c.shutdown()


async def test_saved_plan_does_not_block_direct_independent_recommendation(tmp_path,monkeypatch):
    parent=ScriptedModel(responses=[call('set_research_plan',{'plan': [{'id':'bag','goal':'保留研究计划'}]}),AIMessage(content='已保存'),call('product_search_tool',{'product_id':'P1003'}),rec(),AIMessage(content='结束')])
    c=await _container(tmp_path,monkeypatch,parent)
    try:
        await ask(c,'保存一个研究计划，暂不执行。')
        result,state=await ask(c,'保留原计划，先直接推荐P1003-S1。')
        receipt=next(m for m in reversed(state['messages']) if isinstance(m,ToolMessage) and m.name=='recommend_products')
        print(json.dumps({'receipt':receipt.text,'status':receipt.status,'presentations':len(result.presentations)},ensure_ascii=False))
        assert result.presentations
    finally:await c.shutdown()


async def test_valid_recommendation_without_old_plan_control(tmp_path,monkeypatch):
    parent=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),rec()])
    c=await _container(tmp_path,monkeypatch,parent)
    try:
        result,state=await ask(c,'推荐这款背包，不下单。')
        assert result.error is None and result.status=='completed' and len(result.presentations)==1
    finally:await c.shutdown()


async def test_stale_dependent_research_is_not_reused_in_same_request(tmp_path,monkeypatch):
    from dataclasses import replace
    def child(pid):return ScriptedModel(responses=[call('product_search_tool',{'product_id':pid,'sku_id':pid+'-S1'}),submission(candidates=[{'product_id':pid,'sku_id':pid+'-S1','assessment':'依据本轮前置结果'}])])
    children=[child(pid) for pid in ['P1003','P1008','P1049','P1018']];instances=iter(children)
    def plan(cap):return [{'id':'first','goal':'研究背包','filters':{'price_max_major':cap,'target_currency':'CNY'}},{'id':'second','goal':'根据前置结果研究配套','depends_on':['first']}]
    def dispatch(key):return call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':key}})
    parent=ScriptedModel(responses=[call('set_research_plan',{'plan': plan(200)}),dispatch('first'),dispatch('second'),call('set_research_plan',{'plan': plan(150)}),dispatch('first'),dispatch('second'),AIMessage(content='返回两阶段比较结果')])
    c=await _container(tmp_path,monkeypatch,parent)
    c.orchestrator._sessions._main_factory._settings=replace(c.orchestrator._sessions._main_factory._settings,agent_max_model_rounds=12)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:next(instances))
    try:
        result,state=await ask(c,'先按200元研究，再把第一步改成150元，重新核验前置与依赖结果。')
        batches=[m.artifact['data'] for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch']
        print(json.dumps({'child_calls':[m.cursor for m in children],'tasks':[(b['scope']['step_id'],[c['product_id'] for c in b['candidates']]) for b in batches],'dependent_state':state['task_plan']['tasks']['second'].get('stale')},ensure_ascii=False))
        assert all(m.cursor==2 for m in children),'失效依赖结果被复用，未重新执行依赖研究'
    finally:await c.shutdown()


async def test_parallel_presentations_complete_both_referenced_plan_steps(tmp_path,monkeypatch):
    children=iter([ScriptedModel(responses=[call('product_search_tool',{'product_id':pid,'sku_id':pid+'-S1'}),submission(candidates=[{'product_id':pid,'sku_id':pid+'-S1','assessment':'已读候选'}])]) for pid in ['P1003','P1008']])
    steps=[{'id':'bag','goal':'研究背包'},{'id':'cup','goal':'研究水杯'}]
    parent=ScriptedModel(responses=[call('set_research_plan',{'plan': steps}),AIMessage(content='',tool_calls=[{'name':'task_dispatch','id':step['id'],'args':{'subagent_type':'search_agent','task':{'step_id':step['id']}}} for step in steps]),AIMessage(content='',tool_calls=[{'name':'recommend_products','id':step['id']+'-show','args':{'picks': [{'task_id': step['id'], 'product_id': pid, 'sku_id': pid + '-S1', 'quantity': 1, 'reason': '按本任务依据选择'}], 'guidance': step['goal'] + '结果。','turn_action':'finish'}} for step,pid in zip(steps,['P1003','P1008'])])])
    c=await _container(tmp_path,monkeypatch,parent)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:next(children))
    try:
        result,state=await ask(c,'分别研究背包、水杯，分别交付选择。')
        assert {step['status'] for step in state['task_plan']['steps']}=={'delivered'}
        assert len(result.presentations)==2 and result.status=='completed'
        assert all(p['status']=='completed' for p in result.presentations)
        assert '尚未完成' not in result.final_text and parent.cursor==3
    finally:await c.shutdown()
