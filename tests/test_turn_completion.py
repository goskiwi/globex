"""明确结束意图与真实展示回执共同收口；中间组件不截断本轮。"""
from dataclasses import replace
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from app.application.runtime.delivery import FinalDeliveryMiddleware
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.runtime.results import ToolResult,ToolResultState
from app.application.runtime.tools import as_langchain_tool
from tests.presentation_helpers import delivery_middlewares
from tests.test_agent_handoff import ScriptedModel,call

@pytest.fixture(autouse=True)
def identity():
 token=ShoppingContext.set(replace(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'),source_run_id='r'))
 yield
 ShoppingContext.reset(token)

def data(text):return {'guidance':text,'hits':[{'product_id':'P1003','default_sku_id':'P1003-S1'}]}

async def test_intermediate_details_continue_to_final_comparison_without_extra_summary():
 async def show_product_details():
  """展示中间详情。"""
  return ToolResult(data('先看详情'))
 async def compare_products():
  """交付最终比较。"""
  return ToolResult(data('首选与取舍'))
 model=ScriptedModel(responses=[call('show_product_details',{}),call('compare_products',{},finish=True),AIMessage(content='不应再总结')])
 graph=create_agent(model,tools=[as_langchain_tool(show_product_details),as_langchain_tool(compare_products)],middleware=delivery_middlewares())
 state=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='推荐并比较')]})
 assert model.cursor==2 and state['messages'][-1].content=='首选与取舍'
 assert [p['kind'] for p in state['presentations']]==['product_view','comparison']

async def test_completion_marker_does_not_hide_failed_tool():
 attempts=[]
 async def compare_products():
  """可能失败的比较。"""
  attempts.append(1)
  return ToolResult('来源不足',state=ToolResultState.ERROR,error_code='business_rejected') if len(attempts)==1 else ToolResult(data('已纠正'))
 model=ScriptedModel(responses=[call('compare_products',{},finish=True),call('compare_products',{},finish=True)])
 graph=create_agent(model,tools=[as_langchain_tool(compare_products)],middleware=delivery_middlewares())
 state=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='比较')]})
 assert model.cursor==2 and attempts==[1,1] and state['messages'][-1].content=='已纠正'

async def test_completion_marker_can_follow_last_clean_presentation():
 async def show_product_details():
  """最终查看详情。"""
  return ToolResult(data('只查看这款'))
 model=ScriptedModel(responses=[call('show_product_details',{}),call('finish_turn',{})])
 state=await create_agent(model,tools=[as_langchain_tool(show_product_details)],middleware=delivery_middlewares()).ainvoke({'messages':[HumanMessage(name='buyer',content='查看详情')]})
 assert model.cursor==2 and state['messages'][-1].content=='只查看这款'

async def test_pending_goal_prevents_early_completion(monkeypatch):
 async def show_product_details():
  """只是中间成果。"""
  return ToolResult(data('中间详情'))
 monkeypatch.setattr('app.application.runtime.research_plan.pending_plan_goals',lambda *_:['尚未完成的比较'])
 model=ScriptedModel(responses=[call('show_product_details',{},finish=True),AIMessage(content='已有详情，比较尚未完成。')])
 state=await create_agent(model,tools=[as_langchain_tool(show_product_details)],middleware=delivery_middlewares()).ainvoke({'messages':[HumanMessage(name='buyer',content='查看并比较')]})
 assert model.cursor==2 and state['messages'][-1].content=='已有详情，比较尚未完成。'

async def test_completion_alone_does_not_reuse_failed_previous_batch():
 async def show_product_details():
  """失败的展示。"""
  return ToolResult('失败',state=ToolResultState.ERROR,error_code='business_rejected')
 model=ScriptedModel(responses=[call('show_product_details',{}),call('finish_turn',{}),AIMessage(content='未能交付')])
 state=await create_agent(model,tools=[as_langchain_tool(show_product_details)],middleware=delivery_middlewares()).ainvoke({'messages':[HumanMessage(name='buyer',content='查看')]})
 assert model.cursor==3 and state['messages'][-1].content=='未能交付'
