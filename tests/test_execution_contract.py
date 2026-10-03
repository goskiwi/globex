import json
import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from app.application.runtime.execution import ExecutionMiddleware, graph_step_limit
from tests.native_model_helpers import client_model, completion


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("bad_args", ['{"value":1,,}', '{"value":1', '[]'])
@pytest.mark.parametrize("mixed", [False, True])
async def test_bad_json_is_not_executed_and_can_be_corrected(tmp_path, streamed, bad_args, mixed):
    requests, executed = [], []
    @tool
    def lookup(value: int):
        """读取测试数值。"""
        executed.append(value)
        return str(value)
    def handler(request):
        body=json.loads(request.content); requests.append(body)
        index=len(requests)
        response=completion()
        msg={"role":"assistant", "content":"完成"}
        finish="stop"
        if index < 3:
            msg={"role":"assistant", "content":"", "tool_calls":[{"id":f"c{index}",
                "type":"function", "function":{"name":"lookup", "arguments":bad_args if index==1 else '{"value":2}'}}]}
            finish="tool_calls"
            if index == 1 and mixed:
                msg['tool_calls'].append({'id':'unexecuted','type':'function',
                    'function':{'name':'lookup','arguments':'{"value":99}'}})
        response['choices']=[{'index':0,'message':msg,'finish_reason':finish}]
        if not streamed:
            return httpx.Response(200,json=response)
        delta={k:v for k,v in msg.items() if k!='role'}
        for i,c in enumerate(delta.get('tool_calls',[])): c['index']=i
        chunks=[{**{k:response[k] for k in ('id','model','created')},'object':'chat.completion.chunk',
                 'choices':[{'index':0,'delta':delta,'finish_reason':None}]},
                {'id':'end','model':'fixture','created':1,'object':'chat.completion.chunk',
                 'choices':[{'index':0,'delta':{},'finish_reason':finish}]}]
        return httpx.Response(200,headers={'content-type':'text/event-stream'},
            content=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    model=await client_model(tmp_path,handler,stream=streamed)
    try:
        graph=create_agent(model,tools=[lookup],middleware=[ExecutionMiddleware(5,main=True)])
        result=await graph.ainvoke({'messages':[HumanMessage(content='读取')]},
                                 {'recursion_limit':graph_step_limit(graph,5)})
        assert result['messages'][-1].content=='完成'
        assert executed==[2]
        assert any(m['role']=='tool' and m['tool_call_id']=='c1' for m in requests[1]['messages'])
        original=next(m for m in requests[1]['messages'] if m.get('tool_calls'))
        assert next(c for c in original['tool_calls'] if c['id']=='c1')['function']['arguments']==bad_args
    finally:
        await model.aclose()


@pytest.mark.parametrize("token_budget", [0, 30000])
async def test_main_final_round_disables_tools(tmp_path, token_budget):
    from app.infrastructure.budget import init_budget
    budget = init_budget(token_budget)
    requests=[]
    @tool
    def read():
        """读取资料。"""
        if budget is not None:
            assert budget.delivery_reservation is not None
            assert budget.reserve(budget.remaining + 1) is None
        return '资料'
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        response=completion()
        if len(requests)==1:
            response['choices'][0].update(message={'role':'assistant','content':'','tool_calls':[
                {'id':'read1','type':'function','function':{'name':'read','arguments':'{}'}}]},finish_reason='tool_calls')
        return httpx.Response(200,json=response)
    model=await client_model(tmp_path,handler)
    try:
        graph=create_agent(model,tools=[read],middleware=[ExecutionMiddleware(2,main=True)])
        result=await graph.ainvoke({'messages':[HumanMessage(content='研究')]},
                                  {'recursion_limit':graph_step_limit(graph,2)})
        assert len(requests)==2 and not requests[1].get('tools')
        assert requests[1].get('tool_choice') in (None, 'none')
        assert result['execution_stop']=='model_call_limit'
        if budget is not None:
            assert budget.reserved == 0
    finally:
        await model.aclose()
        init_budget(0)


def test_stop_receipt_is_not_successful_delivery():
    from scripts.eval.interview_assertions import check
    events=[{'type':'final.result','payload':{'text':'已停止','status':'partial','stop_reason':'step_limit'}},
            {'type':'eval.turn.complete','payload':{}}]
    assert not check({'kind':'answer_present'},events)[0]
    assert not check({'kind':'no_execution_error'},events)[0]


async def test_empty_delivery_has_failed_status():
    from tests.test_execution_stop import root
    from tests.test_agent_handoff import ScriptedModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from app.infrastructure.eventbus import TradeEventBus
    graph=create_agent(ScriptedModel(responses=[AIMessage(content='')]),checkpointer=InMemorySaver())
    runner,session=root(graph,TradeEventBus())
    result=await runner._reply('handoff',session,[HumanMessage(content='查询')])
    assert result.status=='failed' and result.stop_reason=='empty_delivery'


def test_invalid_call_and_receipt_cannot_be_split_by_summary():
    from app.application.runtime.context import summary_boundary
    from langchain_core.messages import AIMessage, ToolMessage
    messages=[HumanMessage(content='第一轮'), AIMessage(content='',invalid_tool_calls=[
        {'id':'bad','name':'lookup','args':'{','error':'invalid'}]),
        HumanMessage(content='第二轮'), ToolMessage(content='参数错误',tool_call_id='bad'),
        HumanMessage(content='第三轮'), HumanMessage(content='第四轮')]
    assert summary_boundary(messages,[0,2,4,5],set())==0


def test_shopping_schema_rejects_model_supplied_source_quote():
    from app.application.tools.shopping_state_tool import build_shopping_state_tool
    from pydantic import ValidationError
    tool=build_shopping_state_tool(None)
    exposed=tool.tool_call_schema.model_json_schema()
    assert exposed['required']==['update']
    schema=tool.get_input_schema()
    with pytest.raises(ValidationError) as caught:
        schema.model_validate({'quote':'预算300','update':{'quote':'预算300','goal':'买背包'}})
    locations={tuple(error['loc']) for error in caught.value.errors()}
    assert ('quote',) in locations and ('update','quote') in locations


async def test_timeout_retains_collected_diagnostic_events():
    import asyncio
    from scripts.eval.interview_runtime import LocalCollector
    from app.infrastructure.context_usage import record_evaluation_evidence
    events=[]
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0):
            async with LocalCollector('s','buyer',events=events):
                record_evaluation_evidence('model_argument_error', {'synthetic':True})
                await asyncio.sleep(0)
    assert events[0]['type']=='eval.model_argument_error'
