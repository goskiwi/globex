"""最小购物状态：真实 LangGraph 工具更新、持久恢复及交易数量。"""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.agents.shopping_state import ShoppingWork, Filters, ShoppingUpdate, apply_update, compile_search
from app.application.runtime.working_state import WorkingStateMiddleware
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.shopping_state_tool import build_shopping_state_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.domain.buyer.preference import BuyerPreference, MaterialExclusion
from tests.test_agent_handoff import ScriptedModel
from tests.trade_test_helpers import confirmation_env, test_address as sample_address


@pytest.fixture(autouse=True)
def identity():
    token = ShoppingContext.set(ShoppingContextSnapshot('state','buyer','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)


def change(work, **kwargs):
    work=work.model_copy(update={'source_message_id':'buyer-message'})
    return apply_update(work, ShoppingUpdate(**kwargs), 'buyer-message', ShoppingContext.current().preference_facts)


def call(name, args, id='call'):
    return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':id}])


def test_filters_absent_clear_soft_and_reset_are_distinct():
    work=ShoppingWork(filters=Filters(price_max_major=300,target_currency='CNY',excluded_material_tags=['金属']))
    work=change(work,preferences=['金属更好'],sort='价格升序')
    assert work.filters.excluded_material_tags==['金属']
    assert work.filters.price_max_major==300
    work=change(work,filters={'price_max_major':None})
    assert work.filters.price_max_major is None and work.filters.excluded_material_tags==['金属']
    work=change(work,reset=True,goal='耳机')
    assert work.filters==Filters() and not work.preferences and work.sort is None


def test_explicit_preference_exception_and_restore_one_level():
    prefs=(BuyerPreference('buyer','dislike','不要合成聚合物材质',constraint=MaterialExclusion(('合成聚合物',)),evidence='不要合成聚合物材质'),)
    ShoppingContext.set(replace(ShoppingContext.current(),preference_facts=prefs))
    work=ShoppingWork(filters=Filters())
    assert compile_search(work,prefs)['parameters']['excluded_material_tags']
    work=change(work,ignored_preferences=['不要合成聚合物材质'])
    assert not compile_search(work,prefs)['parameters']['excluded_material_tags']
    work=change(work,ignored_preferences=[])
    assert compile_search(work,prefs)['parameters']['excluded_material_tags']
    with pytest.raises(ValueError):change(work,ignored_preferences=['不存在的偏好'])


def test_selection_by_sku_exclude_and_atomic_failure():
    work=ShoppingWork(filters=Filters())
    work=change(work,selections=[{'product_id':'P1001','sku_id':s,'quantity':q}
                                for s,q in [('P1001-S1',2),('P1001-S2',3)]])
    assert len(work.selections)==2
    work=change(work,excluded_skus=['P1001-S1'])
    assert set(work.selections)=={'P1001-S2'}
    before=work.model_dump()
    with pytest.raises(ValueError):
        change(work,selections=[{'product_id':'P1001','sku_id':'P1001-S1','quantity':1}])
    assert work.model_dump()==before
    work=change(work,excluded_products=['P1001'])
    assert not work.selections


def test_total_budget_is_not_unit_cap_and_missing_currency_unverified():
    work=ShoppingWork(filters=Filters(price_max_major=300),unverified_requirements=['全部商品总预算500元'])
    result=compile_search(work)
    assert result['parameters']['price_max_major'] is None
    assert len(result['unverified_requirements'])==2


@pytest.mark.parametrize('mode',['snapshot','delta'])
async def test_native_update_search_followup_and_checkpoint_restore(tmp_path,mode):
    store=ContextEvidenceStore(tmp_path/'evidence.db')
    search=build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()),TradeEventBus(),store)
    model=ScriptedModel(responses=[
        call('update_shopping_state',{'update':{'filters':{'price_max_major':300,'target_currency':'CNY'}}},'u1'),
        call('product_search_tool',{'normalized_query':'背包'},'s1'),AIMessage(content='完成'),
        call('update_shopping_state',{'update':{'filters':{'price_max_major':None}}},'u2'),
        call('product_search_tool',{'normalized_query':'背包'},'s2'),AIMessage(content='完成'),
        AIMessage(content='当前不限预算')])
    saver=InMemorySaver()
    def build():return create_agent(model,tools=[build_shopping_state_tool(store),as_langchain_tool(search)],
        middleware=[WorkingStateMiddleware(mode)],checkpointer=saver)
    graph=build();config={'configurable':{'thread_id':'state'}}
    first=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='找背包300元')]},config)
    assert first['shopping_work']['filters']['price_max_major']==300
    second=await build().ainvoke({'messages':[HumanMessage(name='buyer',content='不限预算')]},config)
    output=json.loads(next(m.content for m in reversed(second['messages']) if isinstance(m,ToolMessage) and m.name=='product_search_tool'))
    assert output['requirement_application']['parameters']['price_max_major'] is None
    third=await build().ainvoke({'messages':[HumanMessage(name='buyer',content='现在预算是多少？')]},config)
    assert third['shopping_work']['filters']['price_max_major'] is None


async def test_update_cannot_race_search_and_old_state_rejected():
    called=[]
    async def probe()->str:
        """测试只读。"""
        called.append(1);return 'ok'
    model=ScriptedModel(responses=[AIMessage(content='',tool_calls=[
        {'id':'u','name':'update_shopping_state','args':{'update':{'filters':{'price_max_major':300}}}},
        {'id':'s','name':'probe','args':{}}]),AIMessage(content='重试')])
    graph=create_agent(model,tools=[build_shopping_state_tool(None),as_langchain_tool(probe)],middleware=[WorkingStateMiddleware()])
    state=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='300')]})
    assert not called and state['shopping_work']['filters']['price_max_major'] is None
    assert len([m for m in state['messages'] if isinstance(m,ToolMessage) and m.status=='error'])==2
    with pytest.raises(ValueError,match='旧购物状态'):
        await graph.ainvoke({'shopping_work':{'targets':{}},'messages':[HumanMessage(name='buyer',content='继续')]})


async def test_selection_requires_current_session_evidence_and_quantity(tmp_path):
    store=ContextEvidenceStore(tmp_path/'evidence.db')
    await store.save('buyer','state','products',{'hits':[{'product_id':'P1001','skus':[{'sku_id':'P1001-S1'},{'sku_id':'P1001-S2'}]}]})
    choices=[{'product_id':'P1001','sku_id':s,'quantity':q} for s,q in [('P1001-S1',2),('P1001-S2',3)]]
    model=ScriptedModel(responses=[call('update_shopping_state',{'update':{'selections':choices}}),AIMessage(content='已选')])
    graph=create_agent(model,tools=[build_shopping_state_tool(store)],middleware=[WorkingStateMiddleware()])
    result=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='两种都要，各2和3件')]})
    assert result['shopping_work']['selections']['P1001-S1']['quantity']==2
    assert result['shopping_work']['selections']['P1001-S2']['quantity']==3
    model.responses=[call('update_shopping_state',{'update':{'selections':[{'product_id':'P1002','sku_id':'P1002-S1','quantity':1}]}}),AIMessage(content='失败')]
    model.cursor=0
    result=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='要这个')]})
    assert not result['shopping_work']['selections']


async def test_order_graph_reads_saved_quantity_after_checkpoint_restore(confirmation_env):
    from dataclasses import asdict
    from app.application.tools.order_tools import build_create_order_tool
    from app.application.usecases.order_usecases import PlaceOrderUseCase
    env=confirmation_env
    await env.evidence.save('buyer','state','products',{'hits':[{'product_id':'P1001','skus':[{'sku_id':'P1001-S1'}]}]})
    model=ScriptedModel(responses=[
        call('update_shopping_state',{'update':{'selections':[{'product_id':'P1001','sku_id':'P1001-S1','quantity':2}]}}),
        AIMessage(content='已选两件'),
        call('create_order_tool',{'sku_ids':['P1001-S1'],'shipping_address':asdict(sample_address())},'order'),
        AIMessage(content='等待确认')])
    saver=InMemorySaver();config={'configurable':{'thread_id':'state'}}
    def build():return create_agent(model,tools=[build_shopping_state_tool(env.evidence),
        as_langchain_tool(build_create_order_tool(PlaceOrderUseCase(env.service),env.bus,env.evidence))],
        middleware=[WorkingStateMiddleware()],checkpointer=saver)
    await build().ainvoke({'messages':[HumanMessage(name='buyer',content='买两件')]},config)
    before=await env.store.get_inventory()
    result=await build().ainvoke({'messages':[HumanMessage(name='buyer',content='准备确认卡')]},config)
    payload=json.loads(next(m.content for m in result['messages'] if isinstance(m,ToolMessage) and m.name=='create_order_tool'))
    assert payload['confirmation']['payload']['items'][0]['quantity']==2
    assert payload['confirmation']['status']=='pending'
    assert await env.store.get_inventory()==before


@pytest.mark.parametrize('strategy',['legacy','hybrid'])
async def test_exclusions_before_topk_and_default_sku_quote(strategy):
    from app.domain.catalog.product_search_spec import ProductSearchSpec
    repo=InMemoryProductRepository()
    usecase=CatalogSearchUseCase(repo,hybrid_enabled=strategy=='hybrid')
    # 精确查询默认 SKU 被排除时，展示与报价都应来自剩余规格，目录本身不变。
    product=next(p for p in await repo.list_all() if len(p.skus)>1 and all(s.stock>0 for s in p.skus[:2]))
    before=product.primary_available_sku()
    other=product.skus[1]
    result=await usecase.execute(ProductSearchSpec(product_id=product.product_id,ship_to='CN',
        excluded_sku_ids=(before.sku_id,)))
    card=result['hits'][0]
    assert card['default_sku_id']==other.sku_id
    assert card['source_price_major']==other.price.to_major_units()
    assert all(s['sku_id']!=before.sku_id for s in card['skus'])
    assert product.primary_available_sku()==before and card['landed_price']
    # 使用相同召回集核对：被排除的第一名不会占用 top_k。
    query='旅行'
    all_hits=await usecase.execute(ProductSearchSpec(normalized_query=query,top_k=50))
    assert len(all_hits['hits'])>1
    excluded=all_hits['hits'][0]['product_id']
    result=await usecase.execute(ProductSearchSpec(normalized_query=query,top_k=1,excluded_product_ids=(excluded,)))
    assert len(result['hits'])==1 and result['hits'][0]['product_id']!=excluded


async def test_parallel_research_filters_are_local_and_do_not_change_main_state():
    from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
    from app.application.agents.handoff import DelegatedTask
    import asyncio
    seen=[]
    class Child:
        async def ainvoke(self, inputs, **kwargs):
            await asyncio.sleep(0)
            payload=json.loads(inputs['messages'][-1].content)
            effective=ShoppingContext.current().effective_search
            assert payload['parent_context']['effective_search']==effective
            seen.append(effective['parameters']['price_max_major'])
            return {'handoff_result':{'status':'completed','summary':'已核对'}}
    factory=SimpleNamespace(build=lambda:Child())
    dispatch=build_task_dispatch_tool(factory,factory,TradeEventBus())
    original=ShoppingContext.current()
    await asyncio.gather(*(dispatch('search_agent',DelegatedTask(goal='研究',
        filters=Filters(price_max_major=cap,target_currency='CNY'))) for cap in (200,500)))
    assert sorted(seen)==[200,500]
    assert ShoppingContext.current()==original
