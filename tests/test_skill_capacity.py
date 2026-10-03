"""容量判断覆盖实际 Skill 投影，主子角色共享最终请求检查。"""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from app.application.runtime.context import RequestContextMiddleware
from app.application.runtime.skills import SkillReferenceMiddleware
from app.application.runtime.errors import ContextCapacityError
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.capability_tools import build_capability_tools
from app.infrastructure.capability_registry import CapabilityRegistry
from app.infrastructure.buyer_skills import BuyerSkillStore
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_products import token_estimate
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.eventbus import TradeEventBus
from tests.test_retrieval import _settings
from tests.test_agent_handoff import ScriptedModel, factories, call, submission, task


@pytest.fixture
def env(tmp_path):
    registry=CapabilityRegistry(tmp_path/'caps.db')
    personal=BuyerSkillStore(tmp_path/'skills.db')
    token=ShoppingContext.set(ShoppingContextSnapshot('handoff','buyer','zh-CN','CNY',
        capability_digest=registry.bind_session('handoff','buyer')))
    yield SimpleNamespace(registry=registry,personal=personal,path=tmp_path)
    ShoppingContext.reset(token)


def history(skills):
    return [HumanMessage(name='buyer',content='上一轮研究',id='u1'),
        AIMessage(content='',id='a1',tool_calls=[{'id':f'load{i}','name':'load_agent_skill_tool',
            'args':{'skill_id':s['id'],'version':'1'}} for i,s in enumerate(skills)]),
        *[ToolMessage(id=f't{i}',name='load_agent_skill_tool',tool_call_id=f'load{i}',
            content=json.dumps(s,ensure_ascii=False), artifact={"data":s}) for i,s in enumerate(skills)],
        AIMessage(content='上一轮结束',id='a2')]


async def test_inactive_long_skills_do_not_block_small_followup_or_mutate_history(env):
    skills=[env.personal.save('buyer',f'流程{i}','测试流程','原始步骤'*2990) for i in range(9)]
    messages=[*history(skills),HumanMessage(name='buyer',content='你好',id='u2')]
    before=[m.model_dump(mode='json') for m in messages]
    summary=SimpleNamespace(ainvoke=AsyncMock())
    skills_source=SkillReferenceMiddleware(env.registry,env.personal,set())
    context=RequestContextMiddleware(ContextEvidenceStore(env.path/'e.db'),summary,_settings(env.path),
                                     system_prompt="",tools=[],skill_source=skills_source)
    model=ScriptedModel(responses=[AIMessage(content='你好')])
    graph=create_agent(model,middleware=[skills_source,context],
                       checkpointer=InMemorySaver())
    result=await graph.ainvoke({'messages':messages,'read_tool_messages':[f't{i}' for i in range(9)]},
                              {'configurable':{'thread_id':'handoff'}})
    assert token_estimate(before)>113408  # 审查时的默认窗口反例。
    assert result['messages'][-1].text=='你好' and len(model.seen)==1
    assert '原始步骤' not in str(model.seen[0])
    assert result['context_statistics']['before_tokens']<4000
    assert [m.model_dump(mode='json') for m in result['messages'][:-1]]==before
    summary.ainvoke.assert_not_awaited()


async def test_active_bodies_still_count_in_final_request(env):
    skills=[env.personal.save('buyer',f'流程{i}','测试流程','原始步骤'*2990) for i in range(9)]
    messages=history(skills)
    model=ScriptedModel(responses=[AIMessage(content='不应请求模型')])
    skills_source=SkillReferenceMiddleware(env.registry,env.personal,set())
    graph=create_agent(model,middleware=[skills_source,
        RequestContextMiddleware(ContextEvidenceStore(env.path/'e.db'),None,_settings(env.path),
            system_prompt="",tools=[],skill_source=skills_source,summary_enabled=False)])
    with pytest.raises(ContextCapacityError):
        await graph.ainvoke({'messages':messages,'skill_turn_id':'u1',
            'loaded_skills':{s['id']:{k:s[k] for k in ('id','version','content_hash')} for s in skills}})
    assert model.seen==[]


@pytest.mark.parametrize('role',['search','trade'])
@pytest.mark.parametrize('long_body',[False,True])
async def test_real_child_factory_checks_skill_body_before_model(env,monkeypatch,role,long_body):
    skill=env.personal.save('buyer','研究流程','测试流程','原始步骤'*(2990 if long_body else 10))
    factory,model,dispatch,_=factories(env.path,monkeypatch,[
        call('load_agent_skill_tool',{'skill_id':skill['id'],'version':'1'}),
        call('read_candidates',{}),
        submission(candidates=[{'product_id':'P1001','reason':'已读候选'}])],role=role)
    factory._settings=replace(factory._settings,context_size=28000)
    factory.capability_registry=env.registry;factory.buyer_skill_store=env.personal
    result=(await dispatch(role+'_agent',task(skill_refs=[{'id':skill['id'],'version':'1'}]))).data
    if long_body:
        assert result['status']=='failed' and result['stop_reason']=='context_capacity'
        assert result['feedback']==[] and len(model.seen)==1
    else:
        assert result['status']=='completed' and result['stop_reason'] is None
        assert len(model.seen)==3


@pytest.mark.parametrize('large_part',['system','tools'])
async def test_final_guard_counts_system_and_tool_schemas(env,large_part):
    tools=[]
    system='固定规则'
    if large_part=='system':
        system='规则'*18000
    else:
        tools=[StructuredTool.from_function(func=lambda value:value,name='lookup',description='说明'*18000)]
    model=ScriptedModel(responses=[AIMessage(content='不应请求模型')])
    graph=create_agent(model,tools=tools,system_prompt=system,
        middleware=[RequestContextMiddleware(None,None,replace(_settings(env.path),context_size=28000),
            system_prompt=system,tools=tools,summary_enabled=False)])
    with pytest.raises(ContextCapacityError):
        await graph.ainvoke({'messages':[HumanMessage(content='你好')]})
    assert not model.seen


async def test_summary_uses_projected_history_but_archive_keeps_original(env):
    skill=env.personal.save('buyer','早期流程','测试流程','OLD_SKILL_BODY')
    messages=history([skill])
    for i in range(2,7):
        messages.extend([HumanMessage(name='buyer',content='后续请求',id=f'u{i}'),
                         AIMessage(content='后续回答',id=f'a{i+1}')])
    before=[m.model_dump(mode='json') for m in messages]
    from tests.native_context_helpers import summary_selection
    summary=SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=summary_selection())))
    store=ContextEvidenceStore(env.path/'e.db')
    context=RequestContextMiddleware(store,summary,_settings(env.path),system_prompt="",tools=[])
    update=await context.compact_checkpoint({'messages':messages,'read_tool_messages':['t0']},force=True)
    assert 'OLD_SKILL_BODY' not in str(summary.ainvoke.call_args)
    archive=(await store.search('buyer','handoff',kind='context_archive',limit=1))[0]
    assert 'OLD_SKILL_BODY' in json.dumps(archive['data'],ensure_ascii=False)
    assert [m.model_dump(mode='json') for m in messages]==before
    assert update['context_statistics']['status']=='completed'


async def test_capacity_stop_preserves_child_business_evidence(env,monkeypatch):
    factory,model,dispatch,_=factories(env.path,monkeypatch,[
        call('read_candidates',{}),call('load_agent_skill_tool',{'skill_id':'placeholder','version':'1'})])
    skill=env.personal.save('buyer','长流程','测试流程','原始步骤'*2990)
    model.responses[1]=call('load_agent_skill_tool',{'skill_id':skill['id'],'version':'1'})
    factory._settings=replace(factory._settings,context_size=28000)
    factory.capability_registry=env.registry;factory.buyer_skill_store=env.personal
    result=(await dispatch('search_agent',task())).data
    assert result['status']=='partial' and result['stop_reason']=='context_capacity'
    assert result['candidates']==[] and result['observed_product_count']==2
    assert result['evidence_refs']==['ctx_demo'] and len(model.seen)==2
