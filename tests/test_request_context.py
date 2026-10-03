"""固定请求占用先计量，再整理历史，最后发送同一份已校验请求。"""
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
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.capability_registry import CapabilityRegistry
from app.infrastructure.buyer_skills import BuyerSkillStore
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from tests.test_retrieval import _settings
from tests.test_agent_handoff import ScriptedModel, call
from tests.native_context_helpers import summary_selection


@pytest.fixture
def case(tmp_path):
    registry=CapabilityRegistry(tmp_path/'caps.db');personal=BuyerSkillStore(tmp_path/'skills.db')
    skill=personal.save('buyer','流程','比较流程','FLOW_MARKER：'+'核对步骤'*700)
    token=ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY',
        capability_digest=registry.bind_session('s','buyer')))
    messages=[]
    for i in range(8):
        messages.extend([HumanMessage(name='buyer',id=f'u{i}',content='旧请求'),
            AIMessage(id=f'a{i}',content='历史事实'*200)])
    messages.append(HumanMessage(name='buyer',id='current',content='继续当前选购任务'))
    state={'messages':messages,'skill_turn_id':'current','loaded_skills':{
        skill['id']:{k:skill[k] for k in ('id','version','content_hash')}}}
    summary=SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=summary_selection())))
    skills=SkillReferenceMiddleware(registry,personal,set())
    tools=[StructuredTool.from_function(func=lambda value:value,name='lookup',description='读取已知事实')]
    system='固定规则：'+'SYSTEM_RULE '*500
    settings=replace(_settings(tmp_path),context_size=24000,context_target_tokens=24000,context_product_tokens=100000)
    store=ContextEvidenceStore(tmp_path/'evidence.db')
    policy=RequestContextMiddleware(store,summary,settings,system_prompt=system,tools=tools,skill_source=skills)
    yield SimpleNamespace(state=state,policy=policy,summary=summary,skills=skills,skill=skill,
                          tools=tools,system=system,store=store,personal=personal,settings=settings)
    ShoppingContext.reset(token)


def graph_for(case, model):
    return create_agent(model,system_prompt=case.system,tools=case.tools,
        middleware=[case.skills,case.policy],checkpointer=InMemorySaver())


async def test_fixed_overhead_triggers_summary_even_when_history_alone_fits(case):
    model=ScriptedModel(responses=[AIMessage(content='本轮业务回答')])
    original=[m.model_dump(mode='json') for m in case.state['messages']]
    graph=graph_for(case,model)
    result=await graph.ainvoke(case.state,{'configurable':{'thread_id':'s'}})
    stats=result['context_statistics']
    before,after=stats['request_parts_before'],stats['request_parts_after']
    assert before['history_tokens']<before['input_limit']<before['total_tokens']
    assert before['history_budget']==before['input_limit']-before['fixed_tokens']
    assert before['total_tokens']==sum(before[k] for k in (
        'system_tokens','tool_tokens','skill_tokens','state_tokens','other_fixed_tokens','history_tokens','protocol_tokens'))
    assert after['total_tokens']<=after['input_limit']
    assert after['skill_tokens']==before['skill_tokens'] and after['tool_tokens']==before['tool_tokens']
    case.summary.ainvoke.assert_awaited_once()
    assert len(model.seen)==1 and 'FLOW_MARKER' in str(model.seen[0])
    assert result['messages'][-1].content=='本轮业务回答'
    assert sum(isinstance(m,AIMessage) and m.content=='本轮业务回答' for m in result['messages'])==1
    assert [m.model_dump(mode='json') for m in case.state['messages']]==original
    archive=(await case.store.search('buyer','s',kind='context_archive',limit=1))[0]
    assert '历史事实' in json.dumps(archive['data'],ensure_ascii=False)


async def test_fixed_content_alone_cannot_be_fixed_by_deleting_history(case):
    case.system='固定规则'*12000
    model=ScriptedModel(responses=[AIMessage(content='不可发送')])
    graph=graph_for(case,model)
    with pytest.raises(ContextCapacityError,match='固定请求'):
        await graph.ainvoke(case.state,{'configurable':{'thread_id':'s'}})
    assert not model.seen
    case.summary.ainvoke.assert_not_awaited()
    assert not await case.store.search('buyer','s',kind='context_archive',limit=1)


async def test_current_user_and_uncompressible_history_are_not_dropped(case):
    state={'messages':[HumanMessage(name='buyer',id='current',content='当前要求'*5000)]}
    model=ScriptedModel(responses=[AIMessage(content='不可发送')])
    with pytest.raises(ContextCapacityError):
        await graph_for(case,model).ainvoke(state,{'configurable':{'thread_id':'s'}})
    assert not model.seen
    case.summary.ainvoke.assert_not_awaited()
    assert state['messages'][0].text=='当前要求'*5000


async def test_summary_that_does_not_fit_never_reaches_business_model(case):
    case.summary.ainvoke.return_value=AIMessage(content='历史事实'*5000)
    model=ScriptedModel(responses=[AIMessage(content='不可发送')])
    with pytest.raises(ContextCapacityError):
        await graph_for(case,model).ainvoke(case.state,{'configurable':{'thread_id':'s'}})
    assert not model.seen


async def test_skill_revoked_while_summary_runs_is_not_sent(case):
    async def revoke(*args):
        case.personal.delete('buyer',case.skill['id'],'1')
        return AIMessage(content='早期需求已记录')
    case.summary.ainvoke.side_effect=revoke
    model=ScriptedModel(responses=[AIMessage(content='不可发送')])
    with pytest.raises(LookupError):
        await graph_for(case,model).ainvoke(case.state,{'configurable':{'thread_id':'s'}})
    assert not model.seen


async def test_manual_compaction_uses_same_fixed_content_accounting(case):
    manual=await case.policy.compact_checkpoint(case.state)
    model=ScriptedModel(responses=[AIMessage(content='回答')])
    automatic=await graph_for(case,model).ainvoke(case.state,{'configurable':{'thread_id':'s'}})
    assert manual['context_statistics']['request_parts_before']==automatic['context_statistics']['request_parts_before']
    assert manual['context_statistics']['status']=='completed'


async def test_compaction_command_preserves_new_tool_call_and_executes_it_once(case):
    model=ScriptedModel(responses=[call('lookup',{'value':'verified-result'}),AIMessage(content='工具完成后回答')])
    result=await graph_for(case,model).ainvoke(case.state,{'configurable':{'thread_id':'s'}})
    case.summary.ainvoke.assert_awaited_once()
    assert len(model.seen)==2
    receipts=[m for m in result['messages'] if isinstance(m,ToolMessage) and m.name=='lookup']
    assert len(receipts)==1 and receipts[0].content=='verified-result'
    assert result['messages'][-1].content=='工具完成后回答'


def test_removed_paths_are_not_kept_as_aliases():
    import app.application.runtime.context as context
    from app.application.runtime.skills import SkillReferenceMiddleware
    from app.application.runtime.working_state import WorkingStateMiddleware
    from langchain.agents.middleware import AgentMiddleware
    assert not hasattr(context,'EvidenceContextMiddleware')
    assert not hasattr(context,'RequestCapacityMiddleware')
    assert not hasattr(RequestContextMiddleware,'compact')
    assert RequestContextMiddleware.abefore_model is AgentMiddleware.abefore_model
    assert SkillReferenceMiddleware.awrap_model_call is AgentMiddleware.awrap_model_call
    assert WorkingStateMiddleware.awrap_model_call is AgentMiddleware.awrap_model_call
