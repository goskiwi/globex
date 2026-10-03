"""真实工具→有界上下文→原文摘要→实际 usage→报告的贯通反例。"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.runtime.context import RequestContextMiddleware
from app.application.runtime.context_summary import render_summary
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.tool_view import bounded_tool_view
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from app.application.tools.product_details_tool import build_product_details_tool, ProductDetailsInput
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product
from app.domain.catalog.sku import Sku
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_products import token_estimate
from app.infrastructure.context_usage import context_diagnostic_sink, context_usage_sink
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from tests.native_context_helpers import policy, history, summary_selection
from tests.native_model_helpers import client_model, completion
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_execution_stop import root
from tests.test_retrieval import _settings


@pytest.fixture
def scope():
    token = ShoppingContext.set(ShoppingContextSnapshot('handoff', 'buyer', 'zh-CN', 'CNY'))
    yield
    ShoppingContext.reset(token)


@pytest.mark.parametrize('count,window', [(1,128000),(3,24000)])
async def test_unread_large_details_are_bounded_and_next_turn_recovers(tmp_path, scope, count, window):
    description = '\n'.join(f'资料{i:05d}：完整商品信息与不同使用条件需要保留。' for i in range(5000))
    item = Product(product_id='P8800', title='合成资料', brand='测试', category='旅行装备', origin_country='CN',
                   description=description, skus=[Sku('P8800-S1','标准',Money.from_major_units(1500,'CNY'),5)])
    store = ContextEvidenceStore(tmp_path/'evidence.db')
    bus = TradeEventBus()
    detail = build_product_details_tool(CatalogSearchUseCase(InMemoryProductRepository([item])), bus, store)
    tool = as_langchain_tool(detail, args_schema=ProductDetailsInput)
    first = AIMessage(content='',tool_calls=[{'id':f'detail-{i}','name':'get_product_details',
        'args':{'product_id':'P8800'}} for i in range(count)])
    model = ScriptedModel(responses=[first,AIMessage(content='资料已保存，可继续按需读取。'),AIMessage(content='你好')])
    summary = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=summary_selection())))
    settings = replace(_settings(tmp_path),context_size=window)
    middleware = RequestContextMiddleware(store,summary,settings,system_prompt='查看商品',tools=[tool])
    graph = create_agent(model,tools=[tool],middleware=[BusinessToolMiddleware(None,bus),middleware],checkpointer=InMemorySaver())
    runner, session = root(graph,bus)
    result = await runner._reply('handoff',session,[HumanMessage(name='buyer',content='查看资料',id='u1')])
    assert result.status=='completed' and result.stop_reason is None
    state = (await graph.aget_state(session.config)).values
    receipts = [m for m in state['messages'] if isinstance(m,ToolMessage)]
    assert len(receipts)==count
    for receipt in receipts:
        view = json.loads(receipt.content)
        if 'result' in view: view=view['result']  # 中间件提示包装仍计入完整请求。
        assert view['offloaded'] and view['incomplete'] and view['result_ref']
        assert token_estimate(receipt.content)<12000
        original = await store.get('buyer','handoff',view['result_ref'])
        assert original['data']['hits'][0]['description']==description
        lookup = await build_conversation_fact_lookup(store)(result_ref=view['result_ref'],fields='price')
        assert lookup.ok
        assert json.loads(lookup.text)['records'][0]['data']['hits'][0]['skus'][0]['price_major']==1500
    counts=state['context_statistics']['request_parts_after']
    assert counts['total_tokens']<=counts['input_limit']
    assert description not in str(model.seen[1])
    await middleware.compact_checkpoint(state,force=True)
    assert (await runner._reply('handoff',session,[HumanMessage(name='buyer',content='打个招呼',id='u2')])).text=='你好'


async def test_wrapping_fields_are_part_of_view_budget_and_archive_failure_does_not_send_full_text(scope):
    data={'result_ref':'ctx_test','hits':[{'product_id':'P8800','title':'商品','skus':[]}],
          'requirement_application':{'unverified_requirements':['要求'+str(i) for i in range(2000)]}}
    view=await bounded_tool_view(data,object(),ShoppingContext.current(),kind='products',token_limit=512)
    assert token_estimate(view)<=512 and view['offloaded']
    failing=SimpleNamespace(save=AsyncMock(side_effect=OSError('证据存储失败')))
    with pytest.raises(OSError):
        await bounded_tool_view({'text':'长原文'*10000},failing,ShoppingContext.current(),kind='handoff',token_limit=512)


async def test_summary_rejects_rewritten_amount_and_sends_exact_source_after_checkpoint(tmp_path):
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        messages=[]
        for i in range(4):
            messages += [HumanMessage(name='b',id=f'u{i}',content='商品 P1001 的单价为1500元。' if i==0 else '继续讨论。'),
                         AIMessage(id=f'a{i}',content='已阅读。')]
        middleware=policy(tmp_path)
        middleware.model.ainvoke.side_effect=[AIMessage(content='商品 P1001 的单价为150元。'),
                                              AIMessage(content=summary_selection(0))]
        updates=await middleware.compact_checkpoint({'messages':messages},force=True)
        assert [a['status'] for a in updates['context_statistics']['summary_attempts']]==['rejected','accepted']
        assert updates['context_summary']['goals'][0]['text']==messages[0].text
        assert '单价为150元' not in render_summary(updates['context_summary'])
        model=ScriptedModel(responses=[AIMessage(content='查阅原始依据')])
        graph=create_agent(model,middleware=[middleware],checkpointer=InMemorySaver())
        config={'configurable':{'thread_id':'s'}}
        await graph.aupdate_state(config,{'messages':messages},as_node='__start__')
        await graph.aupdate_state(config,updates,as_node='model')
        await graph.ainvoke({'messages':[HumanMessage(name='b',id='new',content='回顾单价')]},config)
        assert any('单价为1500元' in m.text for m in model.seen[-1])
        assert not any('单价为150元' in m.text for m in model.seen[-1])
        archived=await middleware.store.get('b','s',updates['context_summary']['source_ref'])
        assert archived['data']['messages'][0]['content']==messages[0].text
    finally:ShoppingContext.reset(token)


@pytest.mark.parametrize('response', [
    '{"goals":[999],"decisions":[],"open_questions":[],"next_steps":[]}',
    '{"goals":[0],"decisions":[],"open_questions":[],"next_steps":[],"text":"单价150元"}',
    '{"goals":[true],"decisions":[],"open_questions":[],"next_steps":[]}',
])
async def test_invalid_summary_selection_cannot_replace_history(tmp_path,response):
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    try:
        state=history()
        original=[m.model_dump() for m in state['messages']]
        with pytest.raises(ValueError,match='来源'):
            await policy(tmp_path,response=response).compact_checkpoint(state,force=True)
        assert [m.model_dump() for m in state['messages']]==original
    finally:ShoppingContext.reset(token)


def test_incomplete_json_fragment_is_not_promoted_to_a_fact_quote():
    from app.application.runtime.context_summary import source_entries, build_summary
    data={'hits': [], 'fragment': '{"单价":150', 'fragment_format':'json_text',
          'incomplete':True, 'result_ref':'ctx_original'}
    message=ToolMessage(id='large',name='get_product_details',tool_call_id='read',
                        content=json.dumps(data,ensure_ascii=False),artifact={'data':data})
    summary=build_summary(summary_selection(),source_entries([message]),'ctx_archive')
    assert summary['goals'][0]['content_scope']=='reference'
    assert 'ctx_original' in render_summary(summary)
    assert '{"单价":150' not in render_summary(summary)


async def test_actual_context_usage_and_diagnostics_generate_harness_report(tmp_path):
    from scripts.eval.harness.report import render
    from tests.test_harness_evaluation import manifest,rows
    token=ShoppingContext.set(ShoppingContextSnapshot('s','b','zh-CN','CNY'))
    usage=[];diagnostics=[]
    usage_token=context_usage_sink.set(usage.append)
    diagnostic_token=context_diagnostic_sink.set(diagnostics.append)
    model=await client_model(tmp_path,lambda request:httpx.Response(200,json=completion()))
    try:
        middleware=policy(tmp_path,product_tokens=1200)
        graph=create_agent(model,middleware=[middleware])
        await graph.ainvoke(history())
        assert len(usage)==1
        assert usage[0]['request_context']['archived_result_count']==6
        data=rows();data[0]['usage']=usage;data[0]['context_diagnostics']=diagnostics
        report=render(tmp_path,manifest(),data)
        result=report['strategies']['current']['efficiency_diagnostics']
        assert result['archived_results']==6 and result['archive_passes']==1
        assert result['request_statuses']=={'completed':1}
        assert sum(result['estimated_sections'].values())==usage[0]['request_context']['request_parts_after']['total_tokens']
        assert (tmp_path/'report.html').exists()
    finally:
        await model.aclose()
        context_diagnostic_sink.reset(diagnostic_token);context_usage_sink.reset(usage_token);ShoppingContext.reset(token)
