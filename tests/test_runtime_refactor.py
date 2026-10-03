"""重构后的真实边界：Command 回执、最终请求和模型 span。"""
import asyncio
import json
from dataclasses import replace

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.application.runtime.execution import ExecutionMiddleware
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.working_state import WorkingStateMiddleware
from app.application.tools.shopping_state_tool import build_shopping_state_tool
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from tests.native_context_helpers import policy
from tests.native_model_helpers import client_model, completion
from tests.test_agent_handoff import ScriptedModel, call


async def test_command_updates_state_and_emits_one_complete_receipt():
    bus=TradeEventBus();events=[]
    token=ShoppingContext.set(ShoppingContextSnapshot('state','buyer','zh-CN','CNY'))
    model=ScriptedModel(responses=[call('update_shopping_state',{
        'update':{'filters':{'price_max_major':300,'target_currency':'CNY'}}}),AIMessage(content='已记录')])
    graph=create_agent(model,tools=[build_shopping_state_tool(None)],middleware=[
        BusinessToolMiddleware(None,bus),WorkingStateMiddleware()])
    try:
        with observe_run_events(events.append):
            result=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='预算300元')]})
        assert result['shopping_work']['filters']['price_max_major']==300
        receipts=[m for m in result['messages'] if isinstance(m,ToolMessage)]
        assert len(receipts)==1 and receipts[0].status=='success'
        lifecycle=[e.payload['type'] for e in events if e.type=='tool.lifecycle']
        assert lifecycle==['TOOL_CALL_START','TOOL_CALL_DELTA','TOOL_CALL_END',
                           'TOOL_RESULT_START','TOOL_RESULT_TEXT_DELTA','TOOL_RESULT_END']
        assert len([e for e in events if e.type=='tool.result'])==1
        assert 'Command(' not in ''.join(e.payload.get('delta','') for e in events)
    finally:ShoppingContext.reset(token)


async def test_closing_policy_precedes_capacity_validation(tmp_path):
    token=ShoppingContext.set(ShoppingContextSnapshot('state','buyer','zh-CN','CNY'))
    context=policy(tmp_path)
    context.settings=replace(context.settings,context_size=16000)
    declared={'type':'function','function':{'name':'read','description':'资料'*20000,
        'parameters':{'type':'object','properties':{}}}}
    request=ModelRequest(model=ScriptedModel(responses=[]),messages=[HumanMessage(content='收尾')],
        tools=[declared],state={'model_rounds':1,'messages':[HumanMessage(content='收尾')]})
    seen=[]
    async def transport(prepared):
        seen.append(prepared)
        return ModelResponse(result=[AIMessage(content='已完成部分查询')])
    async def prepare(value):return await context.awrap_model_call(value,transport)
    try:
        await ExecutionMiddleware(2,main=True).awrap_model_call(request,prepare)
        assert len(seen)==1 and seen[0].tools==[] and seen[0].tool_choice=='none'
        assert any('最终交付机会' in m.text for m in seen[0].messages)
        assert request.tools==[declared], '准备过程不修改原始声明'
    finally:ShoppingContext.reset(token)


async def test_model_span_covers_transport_and_has_usage_not_zero_duration_sibling(tmp_path,monkeypatch):
    provider=TracerProvider();exporter=InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace,'get_tracer_provider',lambda:provider)
    async def handler(request):
        await asyncio.sleep(.02)
        return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    try:
        await model.ainvoke('合成测试')
        spans=exporter.get_finished_spans()
        models=[s for s in spans if s.name=='globex.model']
        assert len(models)==1 and not any(s.name=='globex.model.attempt' for s in spans)
        span=models[0]
        assert (span.end_time-span.start_time)/1e6>=20
        assert span.attributes['gen_ai.usage.input_tokens']==1500
        assert abs((span.end_time-span.start_time)/1e6-span.attributes['globex.context.elapsed_ms'])<20
    finally:
        await model.aclose()
        provider.shutdown()
