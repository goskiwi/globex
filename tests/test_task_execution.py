"""真实主子图验证任务条件、依赖、交付与短周期收尾；不调用外部模型。"""
import json
from copy import deepcopy
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from tests.test_agent_handoff import ScriptedModel,call,submission
from tests.test_langgraph_runtime import _container
from app.application.usecases.intents import SubmitIntentInput
from app.application.runtime.events import approval_event
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.loops import LoopMiddleware
from app.application.runtime.results import ToolResult
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.tool_view import bounded_tool_view
from app.application.harness.loop_detector import LoopDetector
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus,observe_run_events
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore


def expose_unfiltered_research_source(container, monkeypatch):
    """故意向测试专家开放不适格源，验证下游不会依赖搜索过滤而漏检。"""
    from app.application.tools.product_details_tool import build_product_details_tool
    factory = container.orchestrator._sessions._main_factory._search_factory
    tools = [*factory.build_tools(), as_langchain_tool(build_product_details_tool(
        factory.catalog_search, factory._bus, factory.evidence_store))]
    monkeypatch.setattr(factory, 'build_tools', lambda: tools)


async def ask(c, text, **kwargs):
    result=await c.orchestrator.handle_intent(SubmitIntentInput('s','buyer','zh-CN','CNY',text,**kwargs))
    assert result.error is None,result
    s=c.orchestrator._sessions._agents['s']
    return result,(await s.graph.aget_state(s.config)).values


@pytest.mark.parametrize('partial', [False, True])
async def test_two_research_reports_end_without_cards_or_another_model_call(tmp_path, monkeypatch, partial):
    child_models = iter([ScriptedModel(responses=[
        call('product_search_tool', {'product_id': product}),
        submission(candidates=[{'product_id': product, 'sku_id': product + '-S1',
            'assessment': '适合优先考虑便携性的需求，实际适配仍需核验。', 'caveats': ['是否有电脑仓？']}],unmet_requirements=['必须核验明确附加要求'] if partial and product == 'P1008' else [])]) for product in ('P1003', 'P1008')])
    steps = [{'id':key,'goal':goal,'requirements':['必须核验明确附加要求'] if partial and key=='cup' else []} for key,goal in [('bag', '背包研究'), ('cup', '水杯研究')]]
    model = ScriptedModel(responses=[
        call('set_research_plan', {'plan': steps}),
        AIMessage(content='', tool_calls=[{'name': 'task_dispatch', 'args': {
            'subagent_type': 'search_agent', 'task': {'step_id': step['id']}},
            'id': step['id']} for step in steps]),
        call('present_research', {'reports': [{'task_id': task_id, 'title': {'bag':'背包研究','cup':'水杯研究'}[task_id]} for task_id in ['bag', 'cup']]}, turn_action='finish'),
        AIMessage(content='下一轮无需重新交付研究。')])
    c = await _container(tmp_path, monkeypatch, model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model', lambda *a, **kw: next(child_models))
    try:
        result, state = await ask(c, '分别请专家研究背包和水杯，本轮只要两份研究，不发布最终购物卡')
        assert model.cursor == 3, '研究交付后不能再调用模型收尾'
        assert result.status == ('partial' if partial else 'completed')
        assert result.presentations[-1]["kind"]=="research"
        assert result.final_text.startswith('已交付2份研究报告。')
        assert '背包研究' not in result.final_text and '水杯研究' not in result.final_text
        assert result.presentations[-1]["kind"]=="research"
        payload = next(m.artifact['data'] for m in reversed(state['messages'])
                       if isinstance(m, ToolMessage) and m.name == 'present_research')
        candidates = [c for report in payload['reports'] for c in report['items']]
        assert [report['title'] for report in payload['reports']] == ['背包研究', '水杯研究']
        assert [c['sku_id'] for c in candidates] == ['P1003-S1','P1008-S1']
        assert candidates[0]['facts']['sku']['price_major'] == 129
        assert candidates[0]['facts']['product']['weight_kg'] == .38
        assert candidates[0]['assessment'] == '适合优先考虑便携性的需求，实际适配仍需核验。'
        assert candidates[0]['caveats'] == ['是否有电脑仓？']
        assert 'description' not in result.final_text and '商品原文' not in result.final_text
        assert state['task_plan']['tasks']['bag']['delivery']=='research'
        assert state['task_plan']['tasks']['bag']['delivery']!='products'
        assert not await c.orchestrator._evidence_store.search('buyer', 's', kind='display_batch')
        if not partial:
            followup, _ = await ask(c, '谢谢，暂时不选购')
            assert followup.status == 'completed'
            assert model.cursor == 4, '新轮次不能被旧交付提前结束'
    finally:
        await c.shutdown()


@pytest.mark.parametrize('source', ['missing', 'stale', 'current'])
async def test_research_report_requires_existing_current_task(source):
    from types import SimpleNamespace
    from app.application.tools.research_delivery_tool import build_research_delivery_tool
    from tests.test_research_delivery_boundaries import research_receipt
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    tool = build_research_delivery_tool(ExchangeRateTable())
    report = research_receipt('P1003')
    report['candidates'][0]['facts']['sku'].update(price_major=11958.33, currency='JPY')
    messages = [ToolMessage(content='', name='task_dispatch', tool_call_id='failed',
                            artifact={'data': {'status': 'failed', 'scope': None}}),
                ToolMessage(content='', name='task_dispatch', tool_call_id='valid', artifact={'data': report})]
    tasks = {} if source == 'missing' else {'P1003': {'stale': source == 'stale'}}
    result = await tool([{'task_id':'P1003','title':'背包研究'}], SimpleNamespace(state={'messages': messages, 'task_plan': {'tasks': tasks}}))
    if source == 'current':
        assert result.ok and result.data['status'] == 'completed'
        assert result.data['reports'][0]['items'][0]['facts']['product']['title'] == 'P1003 正确商品名'
        assert result.data['reports'][0]['items'][0]['facts']['sku']['price_major'] == 11958.33, '结构化价格不能失去原币小数'
    else:
        assert not result.ok and result.error_code == 'business_rejected'


def pick(sku='P1003-S1',task_id='bag'):
    return {'product_id':'P1003','sku_id':sku,'quantity':1,'reason':'旅行收纳','task_id':task_id}


def recommend(picks=None,*,bundle=False):
    return call('recommend_bundle' if bundle else 'recommend_products', {'picks': picks or [pick()], 'guidance': '可考虑这款旅行背包。'}, turn_action='continue')


@pytest.mark.parametrize('budget,extra_goal',[(100,False),(150,False),(150,True)])
async def test_plan_verification_and_delivery_share_scoped_conditions(tmp_path,monkeypatch,budget,extra_goal):
    candidate={'product_id':'P1003','sku_id':'P1003-S1','assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}
    child=ScriptedModel(responses=[call('get_product_details',{'product_id':'P1003'}),
                                  submission(candidates=[candidate]),submission(candidates=[candidate]),submission(candidates=[candidate])])
    steps=[{'id':'bag','goal':'选择背包','filters':{'price_max_major':budget}}]
    if extra_goal:steps.append({'id':'cup','goal':'选择水杯','depends_on':['bag']})
    model=ScriptedModel(responses=[*[call('set_shopping_requirements', {'changes': {'filters': {'price_max_major':300,'target_currency':'CNY'}}}), call('set_research_plan', {'plan': steps})],
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}),
        recommend(),AIMessage(content='还有1项计划任务尚未完成' if extra_goal else '有条件尚未满足')])
    c=await _container(tmp_path,monkeypatch,model)
    expose_unfiltered_research_source(c, monkeypatch)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    deliveries=[]
    try:
        with observe_run_events(lambda e:deliveries.append(e.payload['payload']) if e.type=='presentation.result' and e.payload['kind']=='recommendation' else None):
            result,state=await ask(c,'请按分项预算研究商品，不能扩大背包预算')
        dispatch=next((m.artifact or {})['data'] for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch')
        assert dispatch['scope']['effective_search']['parameters']['price_max_major']==budget
        assert dispatch['scope']['effective_search']['parameters']['target_currency']=='CNY'
        assert state['shopping_work']['filters']['price_max_major']==300
        if budget==100:
            assert result.status=='partial'
            assert not deliveries
            assert state['task_plan']['steps'][0]['status']=='blocked'
            assert 'P1003-S1' not in dispatch['qualified_skus']
            assert 'over_price_cap' in dispatch['candidates'][0]['facts']['sku']['constraint_issues']
        else:
            assert len(deliveries)==1 and deliveries[0]['hits'][0]['price_major']==129
            assert result.status==('partial' if extra_goal else 'completed')
            outcome=deliveries[0]['plan_outcome']
            assert outcome['status']=='completed'
            assert state['task_plan']['steps'][0]['status']=='delivered'
            if extra_goal:
                assert '还有1项计划任务尚未完成' in result.final_text
                assert state['task_plan']['steps'][1]['status']=='pending'
    finally:await c.shutdown()


async def test_scoped_bundle_budget_uses_combined_quote(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),submission(candidates=[
        {'product_id':'P1003','sku_id':sku,'assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'} for sku in ['P1003-S1','P1003-S2']])])
    model=ScriptedModel(responses=[*[call('set_shopping_requirements', {'changes': {'filters': {'landed_budget_major':500,'ship_to':'CN','target_currency':'CNY'}}}), call('set_research_plan', {'plan': [{'id':'bag','goal':'组合背包','filters':{'landed_budget_major':200}}]})],
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}),
        recommend([pick(),pick('P1003-S2')],bundle=True),AIMessage(content='组合超过分项预算')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'总预算500，其中背包组合到手总额不超过200')
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='recommend_bundle')
        assert receipt.status=='error' and '所属子任务' in receipt.content
        assert state['task_plan']['steps'][0]['status']=='verified'
    finally:await c.shutdown()


async def test_plan_dependency_blocks_early_dispatch(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[])
    model=ScriptedModel(responses=[call('set_research_plan',{'plan': [
        {'id':'first','goal':'先研究背包'},{'id':'second','goal':'根据背包研究配件','depends_on':['first']}]}),
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'second'}}),
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
        submission(candidates=[{'product_id':'P1003','sku_id':'P1003-S1','assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}])])
    model=ScriptedModel(responses=[*[call('set_shopping_requirements', {'changes': {'filters': {'price_max_major':300,'target_currency':'CNY'}}}), call('set_research_plan', {'plan': [{'id':'bag','goal':'选背包','filters':{'price_max_major':150}}]})],
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}),
        AIMessage(content='已核验候选'),
        call('set_research_plan',{'plan': [{'id':'bag','goal':'选背包','filters':{'price_max_major':100}}]}),
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
        submission(candidates=[{'product_id':'P1003','sku_id':'P1003-S1','assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}])])
    model=ScriptedModel(responses=[call('set_shopping_requirements',{'changes': {'filters':{'price_max_major':300,'target_currency':'CNY'}}}),
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
        submission(candidates=[
            {'product_id':'P1003','sku_id':sku,'assessment':'适合优先考虑便携性的需求，实际适配仍需核验。'}
            for sku in ['P1003-S1','P1003-S2']],caveats=['研究尚未完成'])])
    model=ScriptedModel(responses=[*[call('set_shopping_requirements', {'changes': {'filters': {'price_max_major':300,'target_currency':'CNY'}}}), call('set_research_plan', {'plan': [{'id':'bag','goal':'100元内的背包','filters':{'price_max_major':150 if within_budget else 100}}]})],
        call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}),
        call('compare_products',{'entries':[pick(),pick('P1003-S2')],'preferred_sku_id':None,
                                 'dimensions':[],'guidance':'这两种颜色仅作为对照，研究尚未完成。'}, turn_action='finish'),AIMessage(content='研究仍有缺口')])
    c=await _container(tmp_path,monkeypatch,model)
    expose_unfiltered_research_source(c, monkeypatch)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'找100元内背包；超预算的可以明确作为对照')
        receipt=next(m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='compare_products')
        assert receipt.status=='success'
        data=receipt.artifact['data']
        assert data['plan_outcome']['status']==('completed' if within_budget else 'partial')
        assert all(('over_price_cap' in p['constraint_issues']) is (not within_budget) for p in data['hits'])
        assert state['task_plan']['steps'][0]['status']==('delivered' if within_budget else 'blocked')
        _,resumed=await ask(c,'当前研究完成了吗？')
        assert resumed['task_plan']['steps'][0]['status']==('delivered' if within_budget else 'blocked')
    finally:await c.shutdown()


async def test_duplicate_plan_step_in_parallel_batch_does_not_execute(tmp_path,monkeypatch):
    child=ScriptedModel(responses=[])
    dispatch=call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag'}}).tool_calls[0]
    model=ScriptedModel(responses=[call('set_research_plan',{'plan': [{'id':'bag','goal':'研究背包'}]}),
        AIMessage(content='',tool_calls=[{**dispatch,'id':'a'},{**dispatch,'id':'b'}]),AIMessage(content='需要调整派发')])
    c=await _container(tmp_path,monkeypatch,model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    try:
        _,state=await ask(c,'研究背包')
        receipts=[m for m in state['messages'] if isinstance(m,ToolMessage) and m.name=='task_dispatch']
        assert len(receipts)==2 and all(m.status=='error' and m.artifact['executed'] is False for m in receipts)
        assert child.cursor==0
    finally:await c.shutdown()


def research(step=None):
    task = {'step_id': step} if step is not None else {'goal': '核验旅行背包'}
    return call('task_dispatch', {'subagent_type': 'search_agent', 'task': task})


def research_responses(count):
    return [response for _ in range(count) for response in (
        call('product_search_tool', {'product_id': 'P1003'}),
        submission(candidates=[{'product_id': 'P1003', 'sku_id': 'P1003-S1', 'assessment': '适合优先考虑便携性的需求，实际适配仍需核验。'}]))]


async def test_reset_clears_research_but_preserves_historical_evidence(tmp_path, monkeypatch):
    child = ScriptedModel(responses=research_responses(1))
    model = ScriptedModel(responses=[research(), AIMessage(content='已研究'),
        call('set_shopping_requirements', {'changes': {'reset': True, 'goal': '新的选购任务'}}),
        call('product_search_tool', {'product_id': 'P1003'}),
        recommend([pick(task_id=None)]), AIMessage(content='推荐未完成')])
    c = await _container(tmp_path, monkeypatch, model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model', lambda *a, **kw: child)
    try:
        _, before = await ask(c, '研究旅行背包')
        assert before['task_plan']['tasks']
        receipt = next(m for m in before['messages'] if isinstance(m, ToolMessage) and m.name == 'task_dispatch')
        reference = receipt.artifact['data']['evidence_refs'][0]
        result, after = await ask(c, '重新开始，使用之前看过的背包直接推荐')
        assert result.status == 'completed'
        assert after['task_plan']['tasks'] == {} and after['task_plan']['steps'] == []
        delivered = next(m for m in reversed(after['messages']) if isinstance(m, ToolMessage) and m.name == 'recommend_products')
        assert delivered.status == 'success', delivered.content
        store = ContextEvidenceStore(tmp_path / 'context_evidence.db')
        assert await store.get('buyer', 's', reference) is not None
    finally:
        await c.shutdown()


@pytest.mark.parametrize('change_definition', [True, False])
async def test_dependency_invalidation_survives_turns_and_requires_fresh_research(tmp_path, monkeypatch, change_definition):
    steps = [{'id': 'a', 'goal': '研究背包'},
             {'id': 'b', 'goal': '根据背包研究搭配', 'depends_on': ['a']},
             {'id': 'c', 'goal': '根据搭配进一步核验', 'depends_on': ['b']},
             {'id': 'd', 'goal': '独立研究'}]
    changed = deepcopy(steps)
    changed[0]['goal'] = '重新研究不同用途的背包'
    child = ScriptedModel(responses=research_responses(7))
    update = call('set_research_plan', {'plan': changed}) if change_definition else research('a')
    model = ScriptedModel(responses=[call('set_research_plan', {'plan': steps}),
        research('a'), research('b'), research('c'), research('d'), AIMessage(content='研究完成'),
        update, recommend([pick(task_id='c')]), AIMessage(content='需要重做后续研究'),
        *([research('a')] if change_definition else []), AIMessage(content='前置研究已完成'),
        research('b'), research('c'), AIMessage(content='后续重新核验完成')])
    c = await _container(tmp_path, monkeypatch, model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model', lambda *a, **kw: child)
    try:
        _, state = await ask(c, '按依赖关系研究，并独立研究另一方向')
        assert all(s['status'] == 'verified' for s in state['task_plan']['steps'])
        _, state = await ask(c, '重新研究前置步骤')
        statuses = {s['id']: s['status'] for s in state['task_plan']['steps']}
        assert statuses == {'a': 'pending' if change_definition else 'verified', 'b': 'pending', 'c': 'pending', 'd': 'verified'}
        assert next(m for m in reversed(state['messages']) if isinstance(m, ToolMessage) and m.name == 'recommend_products').status == 'error'
        _, state = await ask(c, '完成前置研究')
        assert [s['status'] for s in state['task_plan']['steps']] == ['verified', 'pending', 'pending', 'verified']
        _, state = await ask(c, '重新核验所有后续步骤')
        assert all(s['status'] == 'verified' for s in state['task_plan']['steps'])
        delegated = [m.content for batch in child.seen for m in batch if getattr(m, 'name', None) == 'delegated_task']
        assert any('dependency_research' in text and 'qualified_skus' in text for text in delegated)
    finally:
        await c.shutdown()


async def test_dependent_research_cannot_rerun_in_same_batch(tmp_path, monkeypatch):
    first, second = research('a'), research('b')
    child = ScriptedModel(responses=research_responses(2))
    model = ScriptedModel(responses=[call('set_research_plan', {'plan': [
        {'id': 'a', 'goal': '前置研究'}, {'id': 'b', 'goal': '后续研究', 'depends_on': ['a']}]}),
        first, second, AIMessage(content='已研究'),
        AIMessage(content='', tool_calls=[{**first.tool_calls[0], 'id': 'retry-a'},
                                         {**second.tool_calls[0], 'id': 'retry-b'}]),
        AIMessage(content='需要顺序执行')])
    c = await _container(tmp_path, monkeypatch, model)
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model', lambda *a, **kw: child)
    try:
        await ask(c, '先研究前置再研究后续')
        _, state = await ask(c, '重做这两项研究')
        assert child.cursor == 4
        rejected = [m for m in state['messages'] if isinstance(m, ToolMessage) and m.tool_call_id.startswith(('retry-a-', 'retry-b-'))]
        assert len(rejected) == 2 and all(m.status == 'error' for m in rejected)
    finally:
        await c.shutdown()
