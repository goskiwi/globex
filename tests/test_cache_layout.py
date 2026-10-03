"""布局在原生 SDK、审批恢复与最终请求边界验证，不请求外部模型。"""
from copy import deepcopy
from dataclasses import replace
import json

import httpx
import pytest
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.application.runtime.working_state import WorkingStateMiddleware
from tests.shopping_state_helpers import work_fixture
from langchain_core.messages import HumanMessage,AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ModelRequest
from app.infrastructure.context_usage import context_usage_sink
from app.infrastructure.prompt_cache import PrefixTracker
from tests.test_prompt_cache import client_model, completion, streaming
from tests.test_harness_evaluation import manifest, rows
from scripts.eval.harness.metrics import summarize
from scripts.eval.harness.contracts import load_suite, ROOT


async def test_native_reply_preserves_static_prefix_latest_budget_and_restore(tmp_path):
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content));return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler);saver=InMemorySaver()
    from tests.native_context_helpers import policy
    def build():return create_agent(model,system_prompt='固定规则',middleware=[
        WorkingStateMiddleware(),policy(tmp_path,system_prompt='固定规则',working_state_mode='snapshot')],checkpointer=saver)
    graph=build();config={'configurable':{'thread_id':'s'}}
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        for budget,text in [(300,'本次预算300元，寄到中国。'),(180,'预算改为180元。')]:
            await graph.ainvoke({'shopping_work':work_fixture(budget=budget),'messages':[HumanMessage(name='b',content=text)]},config)
        assert requests[0]['messages'][0]==requests[1]['messages'][0]
        restored=build()
        state=await restored.aget_state(config)
        assert state.values['shopping_work']['filters']['price_max_major']==180
        assert not any(m.name=='shopping_state' for m in state.values['messages'])
        await restored.ainvoke({'shopping_work':work_fixture(),'messages':[HumanMessage(name='b',content='换个需求，现在想买耳机。')]},config)
        assert (await restored.aget_state(config)).values['shopping_work']['filters']['price_max_major'] is None
    finally:
        ShoppingContext.reset(token);await model.aclose()

async def test_request_projection_anchors_before_tools_without_mutating_history():
    from tests.test_langgraph_runtime import ScriptedModel
    user=HumanMessage(name='b',content='预算180元',id='buyer')
    call=AIMessage(content='',tool_calls=[{'id':'call','name':'order','args':{}}])
    messages=[user,call];before=[m.model_dump() for m in messages]
    work={**work_fixture(budget=180), 'source_message_id':user.id}
    request=ModelRequest(model=ScriptedModel(),messages=messages,tools=[],state={'shopping_work':work},runtime=None)
    from app.application.runtime.working_state import project_working_state
    projected=project_working_state(request.messages,request.state,'snapshot')
    assert projected[0]==user and projected[1].name=='shopping_state'
    assert projected[2]==call
    assert [m.model_dump() for m in messages]==before


def test_prefix_diagnostics_separate_buyer_kind_tools_and_rewrites():
    t=PrefixTracker(capacity=2)
    request={'model':'test','messages':[{'role':'system','content':'规则'}, {'role':'user','content':'私密预算180'}], 'tools':[]}
    original=deepcopy(request)
    first=t.observe(request,scope=('buyer-a','s'),kind='business')
    assert first['prefix_comparison']=='first_request'
    appended=deepcopy(request);appended['messages'].append({'role':'assistant','content':'OK'})
    second=t.observe(appended,scope=('buyer-a','s'),kind='business')
    assert second['prefix_comparison']=='append' and second['prefix_common_messages']==2
    changed=deepcopy(appended);changed['messages'][0]['content']='新规则';changed['tools']=[{'function':'lookup'}]
    third=t.observe(changed,scope=('buyer-a','s'),kind='business')
    assert third['prefix_system_changed'] and third['prefix_tools_changed'] and third['prefix_first_changed_message']==0
    assert t.observe(request,scope=('buyer-b','s'),kind='business')['prefix_comparison']=='first_request'
    assert t.observe(request,scope=('buyer-b','s'),kind='summary')['prefix_comparison']=='first_request'
    assert t.observe(request,scope=None,kind='business')['prefix_comparison']=='first_request'
    assert request==original and '私密预算180' not in repr(t.previous) and 'buyer-a' not in json.dumps(third)
    assert len(t.previous)==2


def test_cache_markers_do_not_masquerade_as_content_changes():
    t=PrefixTracker();plain={'messages':[{'role':'system','content':'规则'}]}
    t.observe(plain,scope=('a','s'),kind='business')
    marked={'messages':[{'role':'system','content':[{'type':'text','text':'规则','cache_control':{'type':'ephemeral'}}]}]}
    assert t.observe(marked,scope=('a','s'),kind='business')['prefix_comparison']=='identical'


def test_cache_layout_suite_only_changes_layout_and_cache():
    suite=load_suite(ROOT/'eval/harness/v1/cache-layout-suite.json')
    strategies=suite['strategies']
    assert len(strategies)==4
    assert {s['overrides']['context_product_tokens'] for s in strategies.values()}=={6000}
    assert {s['overrides']['context_target_tokens'] for s in strategies.values()}=={48000}
    assert suite['gates']['objective']=='prompt_cache'
    assert suite['profiles']['cache_probe']['repetitions']==2
    with pytest.raises(ValueError):WorkingStateMiddleware('typo')


def test_cache_goal_requires_money_not_token_shrinkage():
    m=manifest();m['gates']['objective']='prompt_cache'
    data=rows()
    for row in data:
        row['usage'][0]['input_tokens']=100
        row['usage'][0]['prompt_cache']['reported_cost']=2 if row['strategy']=='current' else 1
    blocked=summarize(m,data)['gates']['candidate']
    assert blocked['status']=='BLOCKED' and not any('输入降幅' in r for r in blocked['reasons'])
    m['pricing']={'verified':True,'currency':'TEST_ONLY','source':'unit_test','verified_at':'fixture'}
    assert summarize(m,data)['gates']['candidate']['status']=='BENEFIT_VERIFIED'
    data[-1]['usage'][0]['prompt_cache']['reported_cost']=None
    assert summarize(m,data)['gates']['candidate']['status']=='BLOCKED'


@pytest.mark.asyncio
@pytest.mark.parametrize('layout',['stable_prefix','legacy_system'])
@pytest.mark.parametrize('scenario',['budget','tools','rebuild'])
async def test_fixed_trace_uses_real_sdk_and_counts_streaming_usage(tmp_path,monkeypatch,layout,scenario):
    from scripts.eval.harness import cache_replay
    requests=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        raw=completion();reply={'budget_major':[300,180,160,140][len(requests)-1],'sku_id':'P1003-S1'}
        chunks=[{'id':'s','object':'chat.completion.chunk','created':1,'model':'fixture','choices':[{'index':0,'delta':{'content':json.dumps(reply)},'finish_reason':'stop'}]},
                {'id':'s','object':'chat.completion.chunk','created':1,'model':'fixture','choices':[],'usage':raw['usage']}]
        return httpx.Response(200,headers={'content-type':'text/event-stream'},content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    model=await client_model(tmp_path,handler,stream=True)
    monkeypatch.setattr(cache_replay,'create_chat_model',lambda *a,**k:model)
    from tests.test_retrieval import _settings
    case={'id':'cache-budget','scenario':scenario,'budgets':[300,180,160,140]}
    row=await cache_replay.run_cache_replay(case,replace(_settings(tmp_path),context_prompt_layout=layout),None,tmp_path,0,None,
        {'output_limit':128,'request_timeout_seconds':5})
    assert row['passed'] and len(row['usage'])==4, json.dumps(row,ensure_ascii=False)
    assert all(u['ttft_ms'] is not None for u in row['usage'])
    assert all(request['tool_choice']=='none' for request in requests)
    assert row['runtime']=='langgraph'
    assert row['usage'][1]['prompt_cache']['prefix_system_changed'] is (layout=='legacy_system')
    assert row['usage'][2]['prompt_cache']['prefix_tools_changed'] is (scenario=='tools')
    assert row['synthetic_summary'] is (scenario=='rebuild')
    if scenario=='rebuild':
        assert row['usage'][2]['replay_stage']=='after_rebuild'
        assert row['usage'][2]['prompt_cache']['prefix_comparison']=='rewrite'
    elif layout=='stable_prefix':
        assert row['usage'][1]['prompt_cache']['prefix_comparison']=='append'
    for request in requests:
        calls=[c['id'] for m in request['messages'] for c in m.get('tool_calls',[])]
        results=[m['tool_call_id'] for m in request['messages'] if m['role']=='tool']
        assert calls==results


@pytest.mark.parametrize('text,fmt,facts',[
    ('{"budget_major":180,"sku_id":"P1003-S1","stock":80}',True,True),
    ('```json\n{"budget_major":180,"sku_id":"P1003-S1"}\n```',True,True),
    ('{"budget_major":180,"sku_id":"P1003-S1"} 额外说明',False,True),
    ('{"budget_major":300,"sku_id":"P1003-S1"}',True,False),
    ('{"budget_major":180,"sku_id":"P1003-S2"}',True,False),
    ('{"budget_major":"180","sku_id":"P1003-S1"}',True,False),
])
def test_replay_facts_and_json_format_have_separate_contracts(text,fmt,facts):
    from scripts.eval.harness.cache_replay import check_replay_answer
    assert check_replay_answer(text,180)=={'json_object':fmt,'budget_and_sku':facts}
