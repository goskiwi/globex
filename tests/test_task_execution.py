"""真实主子图验证任务条件、依赖、交付与短周期收尾；不调用外部模型。"""
import json
from copy import deepcopy
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from tests.test_agent_handoff import ScriptedModel,call,submission
from tests.test_langgraph_runtime import _container
from app.application.agents.orchestrator import SubmitIntentInput
from app.application.runtime.events import approval_event
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.loops import LoopMiddleware
from app.application.runtime.results import ToolResult
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.tool_view import bounded_tool_view
from app.application.harness.loop_detector import LoopDetector
from app.infrastructure.context import ShoppingContext,ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus,observe_run_events
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore


async def ask(c, text, **kwargs):
    result=await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY',text,**kwargs))
    assert result.error is None,result
    s=c.orchestrator._sessions._agents['s']
    return result,(await s.graph.aget_state(s.config)).values


def pick(sku='P1003-S1',task_id='bag'):
    return {'product_id':'P1003','sku_id':sku,'quantity':1,'reason':'旅行收纳','task_id':task_id}


def recommend(picks=None,mode='alternatives'):
    return call('recommend_products',{'picks':picks or [pick()], 'mode':mode,
        'preferred_sku_id':'P1003-S1' if mode=='alternatives' else None,'dimensions':[], 'guidance':'可考虑这款旅行背包。'})


@pytest.mark.parametrize('budget,extra_goal',[(100,False),(150,False),(150,True)])
async def test_plan_verification_and_delivery_share_scoped_conditions(tmp_path,monkeypatch,budget,extra_goal):
    candidate={'product_id':'P1003','sku_id':'P1003-S1','reason':'旅行收纳'}
    child=ScriptedModel(responses=[call('get_product_details',{'product_id':'P1003'}),
                                  submission(candidates=[candidate]),submission(candidates=[candidate]),submission(candidates=[candidate])])
    steps=[{'id':'bag','goal':'选择背包','filters':{'price_max_major':budget}}]
    if extra_goal:steps.append({'id':'cup','goal':'选择水杯','depends_on':['bag']})
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{
        'filters':{'price_max_major':300,'target_currency':'CNY'},'plan':steps}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究背包','step_id':'bag'}}),
        recommend(),AIMessage(content='有条件尚未满足')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    deliveries=[]
    try:
        with observe_run_events(lambda e:deliveries.append(e.payload) if e.type=='recommendation.result' else None):
            result,state=await ask(c,'请按分项预算研究商品，不能扩大背包预算')
        dispatch=next((m.artifact or {})['data'] for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch')
        assert dispatch['scope']['effective_search']['parameters']['price_max_major']==budget
        assert dispatch['scope']['effective_search']['parameters']['target_currency']=='CNY'
        assert state['shopping_work']['filters']['price_max_major']==300
        if budget==100:
            assert result.status=='partial'
            assert not deliveries
            assert state['task_plan']['steps'][0]['status']=='blocked'
            assert any(e['code']=='candidate_constraints_failed' for e in dispatch['feedback'])
        else:
            assert len(deliveries)==1 and deliveries[0]['hits'][0]['price_major']==129
            assert result.status==('partial' if extra_goal else 'completed')
            outcome=deliveries[0]['plan_outcome']
            assert outcome['status']==('partial' if extra_goal else 'completed')
            assert state['task_plan']['steps'][0]['status']=='delivered'
            if extra_goal:
                assert '选择水杯' in deliveries[0]['guidance']
                assert state['task_plan']['steps'][1]['status']=='pending'
    finally:await c.shutdown()


async def test_scoped_bundle_budget_uses_combined_quote(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),submission(candidates=[
        {'product_id':'P1003','sku_id':sku,'reason':'旅行收纳'} for sku in ['P1003-S1','P1003-S2']])])
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{
        'filters':{'landed_budget_major':500,'ship_to':'CN','target_currency':'CNY'},
        'plan':[{'id':'bag','goal':'组合背包','filters':{'landed_budget_major':200}}]}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究背包组合','step_id':'bag'}}),
        recommend([pick(),pick('P1003-S2')],mode='bundle'),AIMessage(content='组合超过分项预算')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'总预算500，其中背包组合到手总额不超过200')
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='recommend_products')
        assert receipt.status=='error' and '所属子任务' in receipt.content
        assert state['task_plan']['steps'][0]['status']=='verified'
    finally:await c.shutdown()


async def test_plan_dependency_blocks_early_dispatch(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[])
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{'plan':[
        {'id':'first','goal':'先研究背包'},{'id':'second','goal':'根据背包研究配件','depends_on':['first']}]}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究配件','step_id':'second'}}),
        AIMessage(content='需要先完成前置研究')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'先研究背包，再根据结果研究配件')
        assert child.cursor==0
        assert next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch').status=='error'
    finally:await c.shutdown()


async def test_ab_warning_and_stop_survive_rejected_approval_and_restart(tmp_path,monkeypatch):
    from tests.test_langgraph_runtime import PreferenceStore
    from app.application.memory.preference_selector import PreferenceSelector
    a=call('get_product_details',{'product_id':'P1003'})
    b=call('get_product_details',{'product_id':'P1001'})
    model=ScriptedModel(responses=[a,b,a,b,call('remember_preference_tool',{'kind':'like','statement':'喜欢轻便'}),
                                  a,b,AIMessage(content='已有资料不足，结束重复查询')])
    store=PreferenceStore()
    def bind(c):
        f=c.orchestrator._sessions._main_factory
        f._preference_store=store;f._preference_selector=PreferenceSelector()
    c=await _container(tmp_path,monkeypatch,model);bind(c)
    try:
        _,state=await ask(c,'核验商品，并申请保存轻便偏好')
        assert state['loop_state'].get('warned') and state['loop_state']['count']==4
        s=c.orchestrator._sessions._agents['s']; event=approval_event((await s.graph.aget_state(s.config)).interrupts)
        await c.shutdown();c=await _container(tmp_path,monkeypatch,model);bind(c)
        result,state=await ask(c,'拒绝',confirmations=({'interrupt_id':event.reply_id+':'+event.tool_calls[0].id,'approved':False},))
        assert result.stop_reason=='repeated_path'
        assert any('研究阶段已结束' in m.text for m in model.seen[-1])
        assert sum(isinstance(m,ToolMessage) and m.name=='get_product_details' for m in state['messages'])==6
        assert state['loop_state']['history']==[]
    finally:await c.shutdown()


async def test_new_user_turn_does_not_inherit_repetition(tmp_path,monkeypatch):
    model=ScriptedModel(responses=[x for _ in range(3) for x in [call('product_search_tool',{'product_id':'P1003'}),AIMessage(content='已核验')]])
    c=await _container(tmp_path,monkeypatch,model)
    try:
        for query in ['查询库存','再次查询当前库存','再实时核验一次']:
            _,state=await ask(c,query)
            assert state['loop_state']['history']==[]
        assert not any(m.name=='loop_feedback' for batch in model.seen for m in batch)
    finally:await c.shutdown()


async def test_complete_business_changes_not_hidden_by_bounded_view(tmp_path):
    store=ContextEvidenceStore(tmp_path/'evidence.db'); policy=LoopDetector(); observations=[]
    async def product_search_tool():
        """合成大结果，尾部库存变化，模型只收到引用。"""
        data={'hits':[{'product_id':f'P{1000+i}','description':'长商品资料'*500,
            'skus':[{'sku_id':f'P{1000+i}-S1','stock':30-len(observations) if i==49 else 80}]} for i in range(50)],
            'recall_strategy':'exact_id_lookup'}
        view=await bounded_tool_view(data,store,ShoppingContext.current(),kind='products',token_limit=240)
        observations.append(view)
        return ToolResult(data,model_data=view)
    model=ScriptedModel(responses=[call('product_search_tool',{}) for _ in range(3)]+[AIMessage(content='结束')])
    graph=create_agent(model,tools=[as_langchain_tool(product_search_tool)],middleware=[BusinessToolMiddleware(policy,TradeEventBus()),LoopMiddleware(policy)])
    token=ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'))
    try:
        result=await graph.ainvoke({'messages':[HumanMessage(content='核验库存')]})
        assert all(v['incomplete'] for v in observations)
        assert not any(m.name=='loop_feedback' for m in result['messages'])
        assert all('返回结构异常' not in m.text for m in result['messages'] if isinstance(m,ToolMessage))
    finally:ShoppingContext.reset(token)


async def test_changed_plan_invalidates_old_qualification(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),
        submission(candidates=[{'product_id':'P1003','sku_id':'P1003-S1','reason':'旅行收纳'}])])
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{
        'filters':{'price_max_major':300,'target_currency':'CNY'},
        'plan':[{'id':'bag','goal':'选背包','filters':{'price_max_major':150}}]}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'选背包','step_id':'bag'}}),
        AIMessage(content='已核验候选'),
        call('update_shopping_state',{'update':{'plan':[{'id':'bag','goal':'选背包','filters':{'price_max_major':100}}]}}),
        recommend(),AIMessage(content='需要按新预算重新核验')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,before=await ask(c,'总预算300，背包150，先研究')
        assert before['task_plan']['steps'][0]['status']=='verified'
        _,after=await ask(c,'背包预算改成100，请推荐')
        receipt=next(m for m in reversed(after['messages']) if isinstance(m,ToolMessage) and m.name=='recommend_products')
        assert receipt.status=='error' and '条件已变化' in receipt.content
        assert after['task_plan']['steps'][0]['status']=='pending'
    finally:await c.shutdown()


async def test_implicit_task_conditions_cannot_be_omitted_at_delivery(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),
        submission(candidates=[{'product_id':'P1003','sku_id':'P1003-S1','reason':'旅行收纳'}])])
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{'filters':{'price_max_major':300,'target_currency':'CNY'}}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究背包','filters':{'price_max_major':150}}}),
        recommend([pick(task_id=None)]),AIMessage(content='需要带上研究来源')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'请单独研究150元以内的背包')
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='recommend_products')
        assert receipt.status=='error' and 'task_id' in receipt.content
    finally:await c.shutdown()


@pytest.mark.parametrize('within_budget',[False,True])
async def test_partial_research_can_be_compared_but_not_claimed_complete(tmp_path,monkeypatch,within_budget):
    child=ScriptedModel(responses=[call('get_product_details',{'product_id':'P1003'}),
        submission(status='partial',unmet_constraints=[] if within_budget else ['超出背包预算'],issues=['研究尚未完成'],candidates=[
            {'product_id':'P1003','sku_id':sku,'reason':'作为超预算对照','unmet_constraints':[] if within_budget else ['超出预算']}
            for sku in ['P1003-S1','P1003-S2']])])
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{
        'filters':{'price_max_major':300,'target_currency':'CNY'},
        'plan':[{'id':'bag','goal':'100元内的背包','filters':{'price_max_major':150 if within_budget else 100}}]}}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究背包','step_id':'bag'}}),
        call('compare_products',{'entries':[pick(),pick('P1003-S2')],'preferred_sku_id':None,
                                 'dimensions':[],'guidance':'这两种颜色仅作为对照，研究尚未完成。'}),AIMessage(content='研究仍有缺口')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'找100元内背包；超预算的可以明确作为对照')
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='compare_products')
        assert receipt.status=='success'
        data=receipt.artifact['data']
        assert data['plan_outcome']['status']=='partial'
        assert all(('over_price_cap' in p['constraint_issues']) is (not within_budget) for p in data['hits'])
        assert state['task_plan']['steps'][0]['status']=='blocked'
        _,resumed=await ask(c,'当前研究完成了吗？')
        assert resumed['task_plan']['steps'][0]['status']=='blocked'
    finally:await c.shutdown()


async def test_duplicate_plan_step_in_parallel_batch_does_not_execute(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[])
    dispatch=call('task_dispatch',{'subagent_type':'search_agent','task':{'goal':'研究','step_id':'bag'}}).tool_calls[0]
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{'plan':[{'id':'bag','goal':'研究背包'}]}}),
        AIMessage(content='',tool_calls=[{**dispatch,'id':'a'},{**dispatch,'id':'b'}]),AIMessage(content='需要调整派发')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'研究背包')
        receipts=[m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch']
        assert len(receipts)==2 and all(m.status=='error' and '本批均未执行' in m.content for m in receipts)
        assert child.cursor==0
    finally:await c.shutdown()
