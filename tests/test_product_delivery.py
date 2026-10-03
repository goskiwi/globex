"""商品区只接收本轮成功交付，不把内部检索候选当作展示批次。"""
import copy
import pytest
from ag_ui.core import RunAgentInput
from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.agents.orchestrator import SubmitIntentInput
from app.infrastructure.eventbus import TradeEvent
from tests.test_agent_handoff import ScriptedModel, call
from tests.test_langgraph_runtime import _container
from langchain_core.messages import AIMessage


def adapter():
    events=[]
    request=RunAgentInput.model_validate({'threadId':'s','runId':'r','messages':[
        {'id':'u','role':'user','content':'找背包'}],'state':{},'tools':[],'context':[],
        'forwardedProps':{'buyerId':'b'}})
    result=AGUIRunAdapter(request,events.append)
    result.start()
    return result,events


def emit(target,kind,payload):
    target.on_trade_event(TradeEvent('s',kind,payload,''))


def test_multiple_searches_never_replace_product_area():
    target,events=adapter()
    for product in ['A','B','C']:
        emit(target,'tool.result',{'tool':'product_search_tool','hits':[{'product_id':product}],
            'recall_strategy':'keyword_2gram'})
    for event in events:
        if event.type=='STATE_SNAPSHOT':
            assert 'products' not in event.snapshot
            assert event.snapshot['recommendation'] is None and event.snapshot['comparison'] is None
    target.finish('目前没有可交付的推荐。','completed',None)
    assert target.state['recommendation'] is None


@pytest.mark.parametrize('kind',['recommendation','comparison'])
def test_only_explicit_successful_delivery_is_published(kind):
    target,events=adapter()
    chosen={'hits':[{'product_id':'B'}]}
    emit(target,kind+'.result',chosen)
    assert target.state[kind] is None, '工具未完成整轮交付前不发布卡片'
    emit(target,'tool.result',{'tool':'product_search_tool','hits':[{'product_id':'C'}]})
    target.finish('已交付','completed',None,product_delivery_complete=True)
    assert target.state[kind]==chosen
    chosen['hits'].clear()
    assert target.state[kind]['hits']==[{'product_id':'B'}]


@pytest.mark.parametrize('status',['failed','partial','cancelled'])
def test_unsuccessful_run_does_not_publish_pending_selection(status):
    target,events=adapter()
    emit(target,'recommendation.result',{'hits':[{'product_id':'A'}]})
    if status=='cancelled':target.fail('已停止',cancelled=True)
    else:target.finish('未完整交付',status,'test-stop')
    assert target.state['recommendation'] is None


async def test_search_only_turn_keeps_evidence_but_has_no_display_batch(tmp_path,monkeypatch):
    model=ScriptedModel(responses=[call('product_search_tool',{'product_id':'P1003'}),
        call('product_search_tool',{'product_id':'P1008'},'second'),AIMessage(content='查询完毕，本轮没有推荐。')])
    container=await _container(tmp_path,monkeypatch,model)
    catalog=container.orchestrator._sessions._main_factory._search_factory._catalog_search
    catalog._embedder=catalog._vector_index=catalog._reranker=None
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','查询商品，不推荐'))
        assert result.error is None
        evidence=container.orchestrator._evidence_store
        assert await evidence.search('b','s',kind='products')
        assert not await evidence.search('b','s',kind='display_batch')
    finally:await container.shutdown()


async def test_followup_carries_committed_cards_and_replacement_archives_once(tmp_path):
    from app.infrastructure.ag_ui_journal import AGUIJournal
    from tests.test_ag_ui_journal import body
    journal=AGUIJournal(tmp_path/'runs.db')
    async def run(identifier,selection=None):
        saved,_=await journal.reserve(body(identifier),'b1','owner')
        request=RunAgentInput.model_validate(saved['input']);events=[]
        target=AGUIRunAdapter(request,events.append,authoritative_state=request.state)
        target.start()
        if selection:
            target.on_trade_event(TradeEvent('s1','recommendation.result',{'hits':[{'product_id':selection}], 'preferred_sku_id':None, 'dimensions':['用途'], 'max_items':12},''))
        target.finish('已回答','completed',None,product_delivery_complete=bool(selection))
        await journal.append(identifier,'owner',[e.model_dump(mode='json',by_alias=True,exclude_none=True) for e in events])
        return target
    await run('r1','A')
    followup=await run('r2')
    assert followup.state['recommendation']['hits']==[{'product_id':'A'}]
    assert followup.state['deliveredRunId']=='r1'
    assert (await journal.session('s1','b1'))['productHistory']==[]
    await run('r3','B');await run('r4')
    restored=await journal.session('s1','b1')
    assert restored['run']['state']['recommendation']['hits']==[{'product_id':'B'}]
    assert restored['run']['state']['deliveredRunId']=='r3'
    assert [r['runId'] for r in restored['productHistory']]==['r1']


@pytest.mark.parametrize('complete',[True,False])
async def test_display_batch_requires_successful_final_delivery(tmp_path,monkeypatch,complete):
    from tests.test_execution_stop import StopModel
    presentation=call('recommend_products',{'mode':'alternatives','guidance':'首选满足这次用途，商品资料与单款理由见卡片。','preferred_sku_id':'P1003-S1','dimensions':['用途'],'picks':[{
        'product_id':'P1003','sku_id':'P1003-S1','quantity':1,'reason':'满足用途','tradeoffs':[]}]},'deliver')
    if not complete:
        presentation.tool_calls.append({'id':'missing-order','name':'query_order_tool','args':{'order_id':'GBX-NOT-FOUND'},'type':'tool_call'})
    model=StopModel(responses=[call('product_search_tool',{'product_id':'P1003'}),presentation])
    container=await _container(tmp_path,monkeypatch,model)
    catalog=container.orchestrator._sessions._main_factory._search_factory._catalog_search
    catalog._embedder=catalog._vector_index=catalog._reranker=None
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY','推荐背包'))
        assert result.status==('completed' if complete else 'partial')
        evidence=container.orchestrator._evidence_store
        assert await evidence.search('b','s',kind='recommendation')
        assert bool(await evidence.search('b','s',kind='display_batch')) is complete
    finally:await container.shutdown()


async def test_limited_main_can_deliver_three_verified_cards_and_preserve_partial_status(tmp_path,monkeypatch):
    from dataclasses import replace
    identifiers=['P1003','P1018','P1049']
    reads=AIMessage(content='',tool_calls=[{'id':pid,'name':'product_search_tool',
        'args':{'product_id':pid}} for pid in identifiers])
    delivery=call('recommend_products',{'mode':'alternatives','preferred_sku_id':'P1003-S1',
        'dimensions':['用途'],'guidance':'已确认这三款，可先比较；第四款尚未确认，没有凑数。',
        'picks':[{'product_id':pid,'sku_id':pid+'-S1','quantity':1,'reason':'本次已确认候选',
                  'tradeoffs':[]} for pid in identifiers]})
    container=await _container(tmp_path,monkeypatch,ScriptedModel(responses=[reads,delivery]))
    factory=container.orchestrator._sessions._main_factory
    factory._settings=replace(factory._settings,agent_max_model_rounds=2)
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput('s','b','zh-CN','CNY',
            '请推荐四款通勤背包，找不齐就说明缺口，不下单。'))
        assert result.status=='partial' and result.stop_reason=='model_call_limit'
        assert result.product_delivery_complete and '第四款尚未确认' in result.final_text
        delivered=await container.orchestrator._evidence_store.search('b','s',kind='display_batch')
        assert len(delivered)==1 and len(delivered[0]['data']['hits'])==3
    finally:
        await container.shutdown()
