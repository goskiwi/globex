"""真实原生校验→工具回执→执行事件→过程摘要，不从错误正文恢复类型。"""
import httpx
import pytest
from ag_ui.core import RunAgentInput
from pydantic import BaseModel

from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from app.application.agents.execution_summary import failure_summary
from app.application.runtime.middleware import BusinessToolMiddleware, ToolResilienceMiddleware
from app.application.runtime.results import ToolResult, ToolResultState, receipt_failure
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.recommendation_tools import RecommendationInput
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus, observe_run_events
from app.infrastructure.resilience import CircuitBreakerRegistry
from tests.native_tool_helpers import tool_graph, call_tool
from tests.test_ag_ui_journal import body


@pytest.fixture
def scope():
    token = ShoppingContext.set(ShoppingContextSnapshot('s1','b1','zh-CN','CNY'))
    yield
    ShoppingContext.reset(token)


async def test_actual_nested_validation_then_corrected_call_retains_two_receipts(scope):
    bus = TradeEventBus()
    adapter = AGUIRunAdapter(RunAgentInput.model_validate(body('validation')),lambda event:None)
    executions = []
    async def recommend_products(picks:list, guidance:str, preferred_sku_id:str|None, dimensions:list, mode:str):
        """交付合成商品，不连接订单或目录。"""
        executions.append(picks)
        payload = {'hits':[{'product_id':p.product_id,'default_sku_id':p.sku_id,'title':'合成颈枕'} for p in picks]}
        bus.publish('s1','recommendation.result',payload)
        return ToolResult(payload)
    tool = as_langchain_tool(recommend_products,args_schema=RecommendationInput)
    graph = tool_graph(tool,middlewares=[BusinessToolMiddleware(None,bus),ToolResilienceMiddleware(CircuitBreakerRegistry())])
    picks=[{'product_id':'P1','sku_id':'P1-S1','quantity':1,'reason':'便携'},
           {'product_id':'P2','sku_id':'P2-S1','reason':'收纳'}]
    arguments={'picks':picks,'guidance':'按需要选择','preferred_sku_id':None,'dimensions':[],'mode':'alternatives'}
    with observe_run_events(adapter.on_trade_event):
        failed = await call_tool(graph,**arguments)
        assert failed.status == 'error' and not executions
        failure=receipt_failure(failed)
        assert failure['code']=='invalid_input' and failure['executed'] is False
        assert failure['issues']==[{'path':['picks',1,'quantity'],'type':'missing','label':'推荐商品 / 第2项 / 购买数量'}]
        first = adapter.state['process']['steps'][0]
        assert first['status']=='failed'
        assert '第2项 / 购买数量：缺少必填字段' in first['summary'] and '本次未执行' in first['summary']
        assert '未确认' not in first['summary']
        picks[1]['quantity']=1
        completed=await call_tool(graph,**arguments)
    assert completed.status=='success' and len(executions)==1
    adapter.finish('已交付','completed',None,product_delivery_complete=True)
    steps=adapter.state['process']['steps']
    assert len(steps)==2 and steps[0]['status']=='failed' and steps[1]['status']=='completed'
    assert steps[0]['id']!=steps[1]['id'] and '已交付2款' in steps[1]['summary']


@pytest.mark.parametrize('kind,expected',[
    ('business_rejected','业务条件未满足'),('unavailable','外部服务'),('internal','工具执行异常')])
async def test_declared_error_category_survives_middleware_event_and_projection(scope,kind,expected):
    bus=TradeEventBus();adapter=AGUIRunAdapter(RunAgentInput.model_validate(body('failure')),lambda e:None)
    async def rejected():
        """合成错误回执。"""
        return ToolResult('timeout 503 库存不足 private address',state=ToolResultState.ERROR,
            error_code=kind,error_reason='库存不足，无法准备确认单' if kind=='business_rejected' else None)
    graph=tool_graph(rejected,middlewares=[BusinessToolMiddleware(None,bus),ToolResilienceMiddleware(CircuitBreakerRegistry())])
    with observe_run_events(adapter.on_trade_event):receipt=await call_tool(graph)
    summary=adapter.state['process']['steps'][0]['summary']
    assert expected in summary and 'private address' not in summary
    assert receipt_failure(receipt)['code']==kind
    if kind=='business_rejected':assert '库存不足' in summary and '未确认' not in summary


async def test_tool_body_validation_error_is_internal_not_input_rejection(scope):
    class InternalRecord(BaseModel):
        value:int
    async def lookup(value:int):
        """参数合法，业务体有程序错误。"""
        InternalRecord(value='private internal value')
    receipt=await call_tool(tool_graph(lookup,middlewares=[ToolResilienceMiddleware(CircuitBreakerRegistry())]),value=1)
    assert receipt_failure(receipt)['code']=='internal'
    assert 'private internal value' not in str(receipt.artifact)


async def test_unexpected_external_failure_is_not_replayed_and_has_safe_reason(scope):
    calls=[]
    async def create_order_tool():
        """请求可能已到达，不能自动重放。"""
        calls.append(1)
        raise httpx.ReadTimeout('private token and address')
    receipt=await call_tool(tool_graph(create_order_tool,middlewares=[ToolResilienceMiddleware(CircuitBreakerRegistry())]))
    failure=receipt_failure(receipt)
    assert calls==[1] and failure['code']=='unavailable'
    assert '未确认' in failure_summary(failure) and 'private' not in str(failure)


async def test_validation_locations_are_structured_without_input_context_or_raw_message(scope):
    async def lookup(quantity:int):
        """只读合成数量。"""
        return ToolResult(quantity)
    receipt=await call_tool(tool_graph(lookup,middlewares=[ToolResilienceMiddleware(CircuitBreakerRegistry())]),quantity='private secret')
    failure=receipt_failure(receipt)
    assert failure['issues']==[{'path':['quantity'],'type':'int_parsing','label':'Quantity'}]
    assert 'private secret' not in str(failure) and '必须填写整数' in failure_summary(failure)


def test_pure_text_error_is_unknown_not_classified_by_words():
    from langchain_core.messages import ToolMessage
    for text in ['参数校验失败：quantity','库存不足','timeout 503']:
        failure=receipt_failure(ToolMessage(content=text,status='error',tool_call_id='unknown'))
        assert failure=={'code':'unknown'}
        assert '未取得明确' in failure_summary(failure)
