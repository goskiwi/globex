"""真实模型、生产中间件与合成只读故障；不接触真实买家或交易。"""
import asyncio
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
import httpx
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from app.application.runtime.results import ToolResult, ToolResultState
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.middleware import runtime_middlewares
from app.application.harness.loop_detector import LoopDetector
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.settings import load_settings
from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.throttle import GatewayThrottle


async def main():
    with tempfile.TemporaryDirectory(prefix='globex-recovery-') as folder:
        settings=replace(load_settings(),data_dir=Path(folder),llm_fallback_model='',llm_max_retries=0,harness_enabled=True)
        model=create_chat_model(settings,stream=False, client=create_chat_client(settings))
        bus=TradeEventBus()
        passed=[]
        try:
            for scenario in ('invalid_input','unavailable'):
                calls=[]
                async def read_catalog(page:int=0):
                    """读取合成目录，page 必须为大于等于 0 的整数。"""
                    calls.append(page)
                    if scenario=='unavailable':
                        raise httpx.ReadTimeout('synthetic service timeout')
                    if page<0:
                        return ToolResult('page 必须大于等于 0，第一页传 0。', state=ToolResultState.ERROR, error_code='invalid_input')
                    return ToolResult('{"hits":[{"product_id":"P1001","sku_id":"P1001-S1"}]}')
                token=ShoppingContext.set(ShoppingContextSnapshot('recovery-'+scenario,'synthetic','zh-CN','CNY'))
                started=time.monotonic()
                try:
                    graph=create_agent(model,tools=[as_langchain_tool(read_catalog)],
                        system_prompt='你是购物助手。参数错误按字段反馈修正；服务暂不可用就说明未完成，不重复故障调用，不编造商品。只回答一句话。',
                        middleware=runtime_middlewares(settings,GatewayThrottle(2,0),CircuitBreakerRegistry(),bus,
                            LoopDetector()))
                    query='测试参数恢复：请先调用 read_catalog(page=-1)，再依据工具反馈纠正参数完成查询。' if scenario=='invalid_input' else '查询第一页商品。'
                    async with asyncio.timeout(60):
                        result=await graph.ainvoke({'messages':[HumanMessage(name='synthetic',content=query)]},
                                                 config={'recursion_limit':12})
                    messages=[m for m in result['messages'] if isinstance(m,ToolMessage)]
                    ok=(calls==[-1,0] and [m.status for m in messages]==['error','success']) if scenario=='invalid_input' else (
                        calls==[0] and len(messages)==1 and messages[0].status=='error')
                    passed.append(ok)
                    print(json.dumps({'scenario':scenario,'passed':ok,'calls':calls,
                        'results':[{'status':m.status,'artifact':m.artifact,'content':m.content} for m in messages],
                        'answer':result['messages'][-1].content,'elapsed_s':round(time.monotonic()-started,3),
                        'model':settings.llm_model,'scope':'真实模型；合成只读工具；无真实交易'},ensure_ascii=False),flush=True)
                except Exception as error:
                    passed.append(False)
                    print(json.dumps({'scenario':scenario,'passed':False,'error_type':type(error).__name__,
                        'elapsed_s':round(time.monotonic()-started,3)}),flush=True)
                finally:
                    ShoppingContext.reset(token)
        finally:
            await model.aclose()
        if not all(passed):
            raise SystemExit(1)


if __name__=='__main__':
    asyncio.run(main())
