"""真实工具批次验证领域增量、并行写入、冲突和checkpoint；不使用字符串合并。"""
from dataclasses import asdict,replace
import json
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.runtime.request_scope import ShoppingContext,ShoppingContextSnapshot
from app.application.runtime.working_state import WorkingStateMiddleware
from app.application.tools.shopping_operations import build_shopping_operations
from app.application.tools.order_tools import build_create_order_tool
from app.application.runtime.tools import as_langchain_tool
from app.application.usecases.order_usecases import PlaceOrderUseCase
from app.application.agents.shopping_state import ShoppingWork,Filters,ShoppingUpdate,apply_update
from app.application.agents.shopping_changes import ShoppingChange,apply_shopping_change
from app.domain.order.address import Address
from tests.test_agent_handoff import ScriptedModel
from tests.trade_test_helpers import confirmation_env
from tests.presentation_helpers import delivery_middlewares

@pytest.fixture(autouse=True)
def identity():
 token=ShoppingContext.set(replace(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'),source_run_id='r'))
 yield
 ShoppingContext.reset(token)

def call(name,args,identifier):return {'id':identifier,'name':name,'args':args}

@pytest.mark.parametrize('different',[False,True])
async def test_select_and_prepare_same_batch_commits_once_or_rejects_real_conflict(confirmation_env,different):
 env=confirmation_env
 await env.evidence.save('buyer','s','products',{'hits':[{'product_id':'P1003','skus':[{'sku_id':'P1003-S1'}]}]})
 item={'product_id':'P1003','sku_id':'P1003-S1','quantity':1}
 address=asdict(Address('合成买家','CN','浙江','杭州','合成测试路1号','310000','13800000000'))
 model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[call('select_products',{'items':[item]},'select'),
  call('create_order_tool',{'items':[{**item,'quantity':2 if different else 1}],'shipping_address':address},'prepare'),call('finish_turn',{},'finish')]),AIMessage(content='选择数量冲突，尚未准备确认')])
 graph=create_agent(model,tools=[*build_shopping_operations(env.evidence),as_langchain_tool(build_create_order_tool(PlaceOrderUseCase(env.service),env.bus,env.evidence))],middleware=[WorkingStateMiddleware(),*delivery_middlewares()],checkpointer=InMemorySaver())
 before=await env.store.get_inventory()
 state=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='保存选择并准备确认')]},{'configurable':{'thread_id':'s'}})
 receipts=[m for m in state['messages'] if isinstance(m,ToolMessage) and m.name!='finish_turn']
 assert [m.status for m in receipts]==(['error','error'] if different else ['success','success'])
 cards=await env.store.list_confirmations(buyer_id='buyer',session_id='s')
 assert len(cards)==(0 if different else 1)
 assert await env.store.get_inventory()==before and (await env.store.list_orders(buyer_id='buyer'))['total']==0
 if not different:
  assert model.cursor==1 and state['shopping_work']['selections']['P1003-S1']['quantity']==1
  assert cards[0]['payload']['items'][0]['quantity']==1

@pytest.mark.parametrize('reverse',[False,True])
async def test_disjoint_fields_and_filter_members_survive_both_call_orders(reverse):
 calls=[call('set_shopping_requirements',{'changes':{'goal':'合成通勤需求','filters':{'ship_to':'CN'}}},'a'),
  call('set_shopping_requirements',{'changes':{'filters':{'price_max_major':200}}},'b'),
  call('set_research_plan',{'plan':[{'id':'bag','goal':'研究背包'}]},'c')]
 model=ScriptedModel(responses=[AIMessage(content='',tool_calls=list(reversed(calls)) if reverse else calls),AIMessage(content='条件已更新')])
 state=await create_agent(model,tools=build_shopping_operations(None),middleware=[WorkingStateMiddleware()]).ainvoke({'messages':[HumanMessage(name='buyer',content='更新需求和计划')]})
 assert all(m.status=='success' for m in state['messages'] if isinstance(m,ToolMessage))
 work=ShoppingWork.model_validate(state['shopping_work'])
 assert work.goal=='合成通勤需求' and work.filters.ship_to=='CN' and work.filters.price_max_major==200 and work.plan[0].id=='bag'

async def test_different_skus_keep_quantities_and_checkpoint(confirmation_env):
 env=confirmation_env
 await env.evidence.save('buyer','s','products',{'hits':[{'product_id':'P1003','skus':[{'sku_id':'P1003-S1'},{'sku_id':'P1003-S2'}]}]})
 calls=[call('select_products',{'items':[{'product_id':'P1003','sku_id':'P1003-S'+str(i),'quantity':i}]},str(i)) for i in [1,2]]
 model=ScriptedModel(responses=[AIMessage(content='',tool_calls=calls),AIMessage(content='已保存'),AIMessage(content='两种数量仍保留')])
 saver=InMemorySaver();config={'configurable':{'thread_id':'s'}}
 def build():return create_agent(model,tools=build_shopping_operations(env.evidence),middleware=[WorkingStateMiddleware()],checkpointer=saver)
 await build().ainvoke({'messages':[HumanMessage(name='buyer',content='两种规格分别1和2件')]},config)
 state=await build().ainvoke({'messages':[HumanMessage(name='buyer',content='看看已选数量')]},config)
 assert {sku:item['quantity'] for sku,item in state['shopping_work']['selections'].items()}=={'P1003-S1':1,'P1003-S2':2}

async def test_reset_cannot_race_selection_or_produce_confirmation(confirmation_env):
 env=confirmation_env
 model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[call('set_shopping_requirements',{'changes':{'reset':True}},'reset'),call('select_products',{'items':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1}]},'select')]),AIMessage(content='需要明确新任务的选择')])
 state=await create_agent(model,tools=build_shopping_operations(env.evidence),middleware=[WorkingStateMiddleware()]).ainvoke({'messages':[HumanMessage(name='buyer',content='新任务')]})
 assert len([m for m in state['messages'] if isinstance(m,ToolMessage) and m.status=='error'])==2
 assert not state['shopping_work']['selections']


def test_delta_preserves_clear_reset_and_rejects_stale_source():
 before=ShoppingWork(filters=Filters(price_max_major=300,ship_to='CN'),goal='旧目标',source_message_id='u',latest_request='合成消息')
 after=apply_update(before,ShoppingUpdate(filters={'price_max_major':None}), 'u')
 patch=ShoppingChange.between(before,after)
 merged=apply_shopping_change(before.model_dump(),patch)
 assert merged['filters']['price_max_major'] is None and merged['filters']['ship_to']=='CN'
 reset=apply_update(before,ShoppingUpdate(reset=True,goal='新目标'),'u')
 result=apply_shopping_change(before.model_dump(),ShoppingChange.between(before,reset,reset=True))
 assert result['goal']=='新目标' and result['filters']==Filters().model_dump() and result['latest_request']=='合成消息'
 with pytest.raises(ValueError,match='绑定当前买家输入'):
  apply_shopping_change({**before.model_dump(),'source_message_id':'new'},patch)


def test_full_snapshot_cannot_replace_existing_state_from_tools():
 work=ShoppingWork(filters=Filters(),source_message_id='u').model_dump()
 assert apply_shopping_change(None,work)==work
 with pytest.raises(ValueError,match='业务工具必须提交领域增量'):
  apply_shopping_change(work,work)


@pytest.mark.parametrize('foreign',[False,True])
async def test_order_source_is_read_once_and_foreign_source_cannot_write(confirmation_env,monkeypatch,foreign):
 from unittest.mock import AsyncMock
 env=confirmation_env
 await env.evidence.save('other' if foreign else 'buyer','s','products',{'hits':[{'product_id':'P1003','skus':[{'sku_id':'P1003-S1'}]}]})
 lookup=AsyncMock(wraps=env.evidence.find_product);monkeypatch.setattr(env.evidence,'find_product',lookup)
 address=asdict(Address('合成买家','CN','浙江','杭州','合成测试路1号','310000','13800000000'))
 model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[call('create_order_tool',{'items':[{'product_id':'P1003','sku_id':'P1003-S1','quantity':1}],'shipping_address':address},'prepare'),call('finish_turn',{},'finish')]),AIMessage(content='来源不符合要求，不能准备确认。')])
 graph=create_agent(model,tools=[as_langchain_tool(build_create_order_tool(PlaceOrderUseCase(env.service),env.bus,env.evidence))],middleware=[WorkingStateMiddleware(),*delivery_middlewares()])
 before=await env.store.get_inventory()
 state=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='准备这一款的确认卡')]})
 assert lookup.await_count==1
 assert len(await env.store.list_confirmations(buyer_id='buyer',session_id='s'))==(0 if foreign else 1)
 assert bool(state['shopping_work']['selections']) is (not foreign)
 assert (await env.store.list_orders(buyer_id='buyer'))['total']==0 and await env.store.get_inventory()==before


@pytest.mark.parametrize('quantity',[True,1.0,'1'])
async def test_invalid_quantity_cannot_be_treated_as_same_write(confirmation_env,quantity):
 env=confirmation_env
 await env.evidence.save('buyer','s','products',{'hits':[{'product_id':'P1003','skus':[{'sku_id':'P1003-S1'}]}]})
 item={'product_id':'P1003','sku_id':'P1003-S1','quantity':1}
 address=asdict(Address('合成买家','CN','浙江','杭州','合成测试路1号','310000','13800000000'))
 model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[call('select_products',{'items':[{**item,'quantity':quantity}]},'select'),call('create_order_tool',{'items':[item],'shipping_address':address},'prepare')]),AIMessage(content='数量不符合要求，请明确数量。')])
 state=await create_agent(model,tools=[*build_shopping_operations(env.evidence),as_langchain_tool(build_create_order_tool(PlaceOrderUseCase(env.service),env.bus,env.evidence))],middleware=[WorkingStateMiddleware()]).ainvoke({'messages':[HumanMessage(name='buyer',content='保存选择并准备确认')]})
 assert not await env.store.list_confirmations(buyer_id='buyer',session_id='s')
 assert not state['shopping_work']['selections']
 assert len([m for m in state['messages'] if isinstance(m,ToolMessage) and m.status=='error'])==2
