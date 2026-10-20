"""计划和独立任务采用互斥输入；依赖双向共用研究模型视图。"""
import json
from types import SimpleNamespace
import pytest
from pydantic import TypeAdapter,ValidationError
from langchain_core.messages import HumanMessage,ToolMessage,AIMessage
from langchain.agents import create_agent
from app.application.agents.handoff import DelegatedTask,PlannedTask,IndependentTask
from app.application.agents.shopping_state import ShoppingWork,Filters,PlanStep,compile_search
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.tool_view import bounded_tool_view,dependency_research_view
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.context_products import token_estimate
from tests.test_agent_handoff import ScriptedModel,call


@pytest.mark.parametrize('field,value',[('filters',None),('filters',{'price_max_major':100}),('requirements',[]),('requirements',['额外要求']),('preferences',[]),('preferences',['另一偏好']),('goal','另一目标')])
def test_plan_input_cannot_redeclare_any_condition(field,value):
    parser=TypeAdapter(DelegatedTask)
    assert isinstance(parser.validate_python({'step_id':'bag'}),PlannedTask)
    with pytest.raises(ValidationError):parser.validate_python({'step_id':'bag',field:value})
    assert isinstance(parser.validate_python({'goal':'独立研究','requirements':['明确要求']}),IndependentTask)


async def test_native_input_rejection_happens_before_child_execution():
    seen=[]
    class Factory:
        def build(self):seen.append(1);raise AssertionError('不应执行子图')
    factory=Factory();tool=as_langchain_tool(build_task_dispatch_tool(factory,factory,TradeEventBus()))
    model=ScriptedModel(responses=[call('task_dispatch',{'subagent_type':'search_agent','task':{'step_id':'bag','requirements':['额外要求']}}),AIMessage(content='需要先更新计划')])
    state=await create_agent(model,tools=[tool]).ainvoke({'messages':[HumanMessage(content='按计划执行')]})
    assert not seen and next(m for m in state['messages'] if isinstance(m,ToolMessage)).status=='error'


@pytest.mark.parametrize('preferences',[None,[],['款式优先']])
async def test_plan_scope_comes_from_saved_definition_and_keeps_parent_state(preferences):
    work=ShoppingWork(filters=Filters(target_currency='CNY'),preferences=['轻便优先'],plan=[PlanStep(id='bag',goal='已保存目标',requirements=['已保存要求'],preferences=preferences)])
    snapshot=ShoppingContextSnapshot('s','b','zh-CN','CNY',effective_search=compile_search(work),source_run_id='r',task_plan={'steps':[{'id':'bag','status':'pending'}],'tasks':{}})
    token=ShoppingContext.set(snapshot);seen=[]
    class Factory:
        def build(self):
            async def invoke(inputs,**kwargs):
                seen.append(json.loads(inputs['messages'][0].content))
                return {'handoff_result':{'status':'needs_input','summary':'需补充','questions':['合成问题']}}
            return SimpleNamespace(ainvoke=invoke)
    try:
        factory=Factory();result=await build_task_dispatch_tool(factory,factory,TradeEventBus())('search_agent',{'step_id':'bag'},runtime=SimpleNamespace(state={'shopping_work':work.model_dump()},tool_call_id='task'))
        assert seen[0]['delegated_task']['goal']=='已保存目标'
        effective=seen[0]['parent_context']['effective_search']
        assert effective['requirements']==['已保存要求'] and effective['preferences']==(['轻便优先'] if preferences is None else preferences)
        assert work.requirements==[] and work.preferences==['轻便优先']
        assert result.data['scope']['definition']['requirements']==['已保存要求']
    finally:ShoppingContext.reset(token)


def research():
    return {'agent':'search_agent','status':'completed','scope':{'task_id':'bag','goal':'背包研究','parent_conditions':{},'definition':{}},
            'candidates':[{'product_id':'P1003','sku_id':'P1003-S1','assessment':'已选候选','facts':{'result_ref':'source-ref','product':{'title':'合成背包'},'sku':{'sku_id':'P1003-S1'}}}],
            'observed_products':[f'P{i}' for i in range(1000,1272)],'observed_skus':{f'P{i}':[f'P{i}-S{j}' for j in range(1,9)] for i in range(1000,1272)},'evidence_refs':['source-ref']}


async def test_dependencies_share_parent_model_projection_and_combined_budget():
    raw=research();before=json.dumps(raw,ensure_ascii=False)
    parent=await bounded_tool_view(raw,None,None,kind='handoff',token_limit=12000)
    dependencies=await dependency_research_view({'bag':raw,'second':raw},None,None,token_limit=12000)
    assert dependencies['bag']==dependencies['second']==parent
    assert 'observed_skus' not in parent and 'parent_conditions' not in parent['scope']
    assert parent['candidates'][0]['facts']['result_ref']=='source-ref'
    assert token_estimate(dependencies)<12000 and token_estimate(parent)<token_estimate(raw)
    assert json.dumps(raw,ensure_ascii=False)==before


async def test_small_dependency_view_keeps_readable_full_evidence(tmp_path):
    from copy import deepcopy
    from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
    from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
    raw=research();store=ContextEvidenceStore(tmp_path/'evidence.db')
    candidate=raw['candidates'][0]
    raw['candidates']=[]
    for i in range(1000,1008):
        item=deepcopy(candidate);item.update(product_id=f'P{i}',sku_id=f'P{i}-S1')
        item['facts']['sku']['sku_id']=item['sku_id'];raw['candidates'].append(item)
    context=ShoppingContextSnapshot('s','b','zh-CN','CNY');token=ShoppingContext.set(context)
    try:
        view=(await dependency_research_view({'bag':raw},store,context,token_limit=500))['bag']
        assert view['offloaded'] and view['status']=='completed'
        lookup=build_conversation_fact_lookup(store);offset=0;parts=[]
        while True:
            result=await lookup(result_ref=view['result_ref'],field_offset=offset)
            assert result.ok
            fragment=result.data['records'][0]['data'];parts.append(fragment['excerpt'])
            if fragment['next_field_offset'] is None:break
            offset=fragment['next_field_offset']
        assert json.loads(''.join(parts))==raw
        assert 'result_ref' not in raw
    finally:ShoppingContext.reset(token)
