"""原生图的渐进式 Skill 加载、子任务隔离和压缩后恢复。"""
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
from ag_ui.core import RunAgentInput
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.runtime.skills import SkillReferenceMiddleware
from app.application.runtime.context import RequestContextMiddleware
from tests.test_retrieval import _settings
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.capability_tools import build_capability_tools, CAPABILITY_POLICY
from app.application.agents.skill_access import read_skill
from app.infrastructure.buyer_skills import BuyerSkillStore
from app.infrastructure.capability_registry import CapabilityRegistry
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.ag_ui_journal import AGUIJournal, JournalConflict
from app.presentation.ag_ui import parse_intent
from tests.test_agent_handoff import ScriptedModel, call, task, submission, factories
from tests.test_capability_registry import publish, document
from tests.test_ag_ui import request_data


@pytest.fixture
def env(tmp_path):
    registry=CapabilityRegistry(tmp_path/'registry.db')
    personal=BuyerSkillStore(tmp_path/'personal.db')
    skill=personal.save('buyer','通勤背包流程','核对通勤背包需求','PRIVATE_BODY：先核对重量，再比较。')
    token=ShoppingContext.set(ShoppingContextSnapshot('handoff','buyer','zh-CN','CNY',
        capability_digest=registry.bind_session('handoff','buyer')))
    yield SimpleNamespace(registry=registry,personal=personal,skill=skill,path=tmp_path)
    ShoppingContext.reset(token)


def graph_for(env,model,saver=None):
    tools=[as_langchain_tool(fn) for fn in build_capability_tools(env.registry,set(),TradeEventBus(),env.personal)]
    skills=SkillReferenceMiddleware(env.registry,env.personal,set())
    return create_agent(model,tools=tools,system_prompt=CAPABILITY_POLICY,
        middleware=[skills,RequestContextMiddleware(None,None,_settings(env.path),
            system_prompt=CAPABILITY_POLICY,tools=tools,skill_source=skills,summary_enabled=False)],
        checkpointer=saver or InMemorySaver())


def load_call(skill):
    return call('load_agent_skill_tool',{'skill_id':skill['id'],'version':skill['version']})


async def test_directory_then_load_then_body_and_next_turn_reset(env):
    other=env.personal.save('buyer','耳机流程','耳机需求','UNRELATED_BODY')
    model=ScriptedModel(responses=[load_call(env.skill),AIMessage(content='按流程工作'),AIMessage(content='查订单')])
    graph=graph_for(env,model);config={'configurable':{'thread_id':'handoff'}}
    result=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='通勤背包')]},config)
    assert 'PRIVATE_BODY' not in str(model.seen[0])
    assert 'PRIVATE_BODY' in str(model.seen[1]) and 'UNRELATED_BODY' not in str(model.seen[1])
    assert set(result['loaded_skills'])=={env.skill['id']}
    assert len([m for m in model.seen[1] if m.name=='skill_reference'])==1
    assert 'body' not in result['loaded_skills'][env.skill['id']]
    result=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='查订单')]},config)
    assert result['loaded_skills']=={} and 'PRIVATE_BODY' not in str(model.seen[2])


async def test_parallel_loads_do_not_overwrite_each_other(env):
    second=env.personal.save('buyer','第二流程','比较','SECOND_BODY')
    model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[
        {'id':'one','name':'load_agent_skill_tool','args':{'skill_id':env.skill['id'],'version':'1'}},
        {'id':'two','name':'load_agent_skill_tool','args':{'skill_id':second['id'],'version':'1'}}]),AIMessage(content='完成')])
    result=await graph_for(env,model).ainvoke({'messages':[HumanMessage(name='buyer',content='比较两类')]},
                                            {'configurable':{'thread_id':'handoff'}})
    assert len(result['loaded_skills'])==2
    assert 'PRIVATE_BODY' in str(model.seen[1]) and 'SECOND_BODY' in str(model.seen[1])


async def test_archived_tool_body_is_rehydrated_from_refs_after_graph_rebuild(env):
    # 在用户尚未收到回答前暂停，模拟上下文整理与进程重建。
    saver=InMemorySaver();model=ScriptedModel(responses=[load_call(env.skill)])
    tools=[as_langchain_tool(f) for f in build_capability_tools(env.registry,set(),TradeEventBus(),env.personal)]
    graph=create_agent(model,tools=tools,middleware=[SkillReferenceMiddleware(env.registry,env.personal,set())],
        checkpointer=saver,interrupt_before=['model'])
    config={'configurable':{'thread_id':'handoff'}}
    await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='通勤选购')]},config)
    await graph.ainvoke(None,config)  # 执行加载工具，再次停在模型前；before_model 已记录引用。
    snapshot=await graph.aget_state(config)
    assert env.skill['id'] in snapshot.values['loaded_skills']
    original=next(m for m in snapshot.values['messages'] if isinstance(m,ToolMessage))
    archived=original.model_copy(update={'content':'{"archived":true,"notice":"正文已归档"}'})
    await graph.aupdate_state(config,{'messages':[archived]},as_node='tools')
    resumed=ScriptedModel(responses=[AIMessage(content='按原流程继续')])
    restored=graph_for(env,resumed,saver)
    await restored.ainvoke(None,config)
    assert 'PRIVATE_BODY' in str(resumed.seen[0])
    assert env.skill['id'] in (await restored.aget_state(config)).values['loaded_skills']


async def test_revoked_or_other_buyer_skill_cannot_be_used(env):
    other=env.personal.save('someone-else','私有','私有','SECRET_OTHER_BUYER')
    with pytest.raises(LookupError):read_skill(env.registry,env.personal,set(),other['id'],'1')
    env.personal.delete('buyer',env.skill['id'],'1')
    model=ScriptedModel(responses=[load_call(env.skill),AIMessage(content='资料不可用')])
    result=await graph_for(env,model).ainvoke({'messages':[HumanMessage(name='buyer',content='选购')]},
                                             {'configurable':{'thread_id':'handoff'}})
    assert not result['loaded_skills']
    assert 'PRIVATE_BODY' not in str(model.seen[-1]) and 'SECRET_OTHER_BUYER' not in str(model.seen)


async def test_child_loads_own_reference_without_parent_history(env,tmp_path,monkeypatch):
    search,_,dispatch,bus=factories(tmp_path,monkeypatch,[])
    search.capability_registry=env.registry;search.buyer_skill_store=env.personal
    child=ScriptedModel(responses=[load_call(env.skill),call('read_candidates',{}),
        submission(candidates=[{'product_id':'P1001','reason':'候选'}])])
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**k:child)
    out=(await dispatch('search_agent',task(skill_refs=[{'id':env.skill['id'],'version':'1'}]))).data
    assert out['status']=='completed'
    assert 'PRIVATE_BODY' not in str(child.seen[0]) and 'PRIVATE_BODY' in str(child.seen[1])
    assert any(m.name=='delegated_task' for m in child.seen[0])
    assert not any(m.name=='buyer' for m in child.seen[0])


@pytest.mark.parametrize('value',[None,{}, {'id':'x','version':'1','contentHash':'a'*64}])
def test_removed_selected_skill_request_is_rejected(value):
    body=request_data(forwardedProps={'buyerId':'buyer','selectedSkill':value})
    with pytest.raises(HTTPException) as error:parse_intent(RunAgentInput.model_validate(body))
    assert error.value.status_code==422


def test_role_tool_scope_is_not_expanded(env):
    publish(env.registry,document())
    # 使用新会话读取已发布目录，交易角色不能加载要求搜索工具的流程。
    token=ShoppingContext.set(ShoppingContextSnapshot('new','buyer','zh-CN','CNY'))
    try:
        with pytest.raises(ValueError):
            read_skill(env.registry,env.personal,{'query_order_tool'},'backpack','1')
        assert read_skill(env.registry,env.personal,{'product_search_tool'},'backpack','1')['body']
    finally:ShoppingContext.reset(token)


async def test_real_compaction_and_reopened_evidence_are_readable_by_child(env,tmp_path,monkeypatch):
    from tests.native_context_helpers import history, policy
    from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
    from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
    state=history()
    for message in state['messages']:
        if isinstance(message,HumanMessage):message.name='buyer'
    middleware=policy(tmp_path,product_tokens=1)
    updates=await middleware.compact_checkpoint(state)
    archived=next(m for m in updates['messages'] if isinstance(m,ToolMessage) and m.id=='t0')
    ref=json.loads(archived.content)['result_ref']
    reopened=ContextEvidenceStore(tmp_path/'e.db')
    assert (await reopened.get('buyer','handoff',ref))['data']['hits'][0]['product_id']=='P0'
    tool=as_langchain_tool(build_conversation_fact_lookup(reopened))
    search,_,dispatch,_=factories(tmp_path,monkeypatch,[],tools=[tool])
    search.capability_registry=env.registry;search.buyer_skill_store=env.personal
    child=ScriptedModel(responses=[load_call(env.skill),call('conversation_fact_lookup',{'result_ref':ref}),
        submission(status='partial',summary='已回查历史，尚需当前核验',
            candidates=[{'product_id':'P0','reason':'历史已读对象'}],
            unmet_constraints=['尚未重新核验当前库存'])])
    monkeypatch.setattr('app.application.agents.search_agent.create_chat_model',lambda *a,**kw:child)
    output=(await dispatch('search_agent',task(
        skill_refs=[{'id':env.skill['id'],'version':'1'}]))).data
    assert output['status']=='partial' and output['historical_evidence_refs']==[ref]
    assert '商品限制'*10 not in str(child.seen[0])
    assert 'PRIVATE_BODY' in str(child.seen[1])
    assert 'P0' in str(child.seen[2])


async def test_expired_active_revision_is_rejected_before_next_model_call(env):
    model=ScriptedModel(responses=[load_call(env.skill),AIMessage(content='完成')])
    graph=graph_for(env,model)
    state=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='通勤')]},
                             {'configurable':{'thread_id':'handoff'}})
    env.personal.delete('buyer',env.skill['id'],'1')
    from langchain.agents.middleware import ModelRequest
    observed=[]
    async def handler(request):observed.append(request)
    with pytest.raises(LookupError):
        await SkillReferenceMiddleware(env.registry,env.personal,set()).request_parts(state)
    assert observed==[]


async def test_active_refs_restore_from_sqlite_without_replaying_load(env,tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    path=str(tmp_path/'graph.db');config={'configurable':{'thread_id':'handoff'}}
    tools=[as_langchain_tool(f) for f in build_capability_tools(env.registry,set(),TradeEventBus(),env.personal)]
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph=create_agent(ScriptedModel(responses=[load_call(env.skill)]),tools=tools,
            middleware=[SkillReferenceMiddleware(env.registry,env.personal,set())],
            checkpointer=saver,interrupt_before=['model'])
        await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='通勤选购')]},config)
        await graph.ainvoke(None,config)
        assert env.skill['id'] in (await graph.aget_state(config)).values['loaded_skills']
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        model=ScriptedModel(responses=[AIMessage(content='恢复后继续')])
        restored=graph_for(env,model,saver)
        result=await restored.ainvoke(None,config)
        assert 'PRIVATE_BODY' in str(model.seen[0])
        assert sum(isinstance(m,ToolMessage) and m.name=='load_agent_skill_tool' for m in result['messages'])==1


async def test_journal_does_not_accept_legacy_manual_execution(tmp_path):
    journal=AGUIJournal(tmp_path/'runs.db')
    body=request_data(forwardedProps={'buyerId':'buyer','selectedSkill':{'id':'x','version':'1'}})
    with pytest.raises(JournalConflict,match='selectedSkill'):
        await journal.reserve(body,'buyer','owner')
