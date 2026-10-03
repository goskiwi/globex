"""真实模型读取合成商品后，受控注入预算耗尽；验证主/子任务交付，不触碰交易。

test_stop 是故障注入，不代表实际供应商 token 消耗。
"""
import asyncio
import json
import tempfile
import time
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.agents.handoff import build_submission_tool, HANDOFF_POLICY
from app.application.runtime.handoff import HandoffMiddleware, HandoffContext
from app.application.runtime.middleware import runtime_middlewares
from app.application.runtime.tools import as_langchain_tool
from app.application.runtime.results import ToolResult
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.application.harness.loop_detector import LoopDetector
from app.infrastructure.budget import init_budget, get_budget
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_usage import context_usage_sink
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.settings import load_settings
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.throttle import GatewayThrottle


async def main():
    reports=[]
    with tempfile.TemporaryDirectory(prefix='globex-stop-live-') as folder:
        settings=replace(load_settings(),data_dir=Path(folder),llm_fallback_model='',llm_max_retries=0,harness_enabled=True)
        model=create_chat_model(settings,stream=False, client=create_chat_client(settings))
        for role in ('main','child'):
            started=time.monotonic();usage=[];calls=[]
            bus=TradeEventBus()
            token=ShoppingContext.set(ShoppingContextSnapshot('stop-live','synthetic','zh-CN','CNY'))
            usage_token=context_usage_sink.set(usage.append)
            init_budget(1_000_000)
            async def product_search_tool():
                """读取两件合成商品，先单独调用，再进行总结或提交。"""
                calls.append('search')
                data={'hits':[{'product_id':'P1001','skus':[{'sku_id':'P1001-S1'}]},
                              {'product_id':'P1002','skus':[{'sku_id':'P1002-S1'}]}],
                      'result_ref':'ctx_synthetic','recall_strategy':'synthetic_fixture'}
                bus.publish('stop-live','tool.result',{'tool':'product_search_tool',**data})
                budget=get_budget();budget.charge('test_stop',budget.remaining)
                return ToolResult(data)
            middlewares=runtime_middlewares(settings,GatewayThrottle(2,0),CircuitBreakerRegistry(),bus,
                                            LoopDetector())
            prompt='这是合成购物测试。先单独调用 product_search_tool 读取商品，不要在同一轮提交或调用其他工具，然后再完成总结。'
            try:
                async with asyncio.timeout(60):
                    if role=='main':
                        graph=create_agent(model,tools=[as_langchain_tool(product_search_tool)],system_prompt=prompt,
                            middleware=middlewares,checkpointer=InMemorySaver())
                        runner=MainAgentOrchestrator.__new__(MainAgentOrchestrator)
                        runner._bus=bus;runner._native_observer=ContextVar('stop-live-observer',default=None)
                        session=SimpleNamespace(graph=graph,config={'configurable':{'thread_id':'stop-live'}})
                        output=await runner._reply('stop-live',session,[HumanMessage(name='synthetic',content='读取合成商品并总结')])
                        snapshot=await graph.aget_state(session.config)
                        passed=(output.status=='partial' and 'ctx_synthetic' in output.text
                                and '预算不足' in output.text and snapshot.next==())
                        output={'text':output.text,'status':output.status,'stop_reason':output.stop_reason}
                    else:
                        graph=create_agent(model,tools=[as_langchain_tool(product_search_tool),build_submission_tool()],
                            system_prompt=prompt+HANDOFF_POLICY,middleware=[HandoffMiddleware(),*middlewares],
                            context_schema=HandoffContext)
                        factory=SimpleNamespace(build=lambda:graph)
                        dispatch=build_task_dispatch_tool(factory,factory,bus)
                        output=(await dispatch('search_agent',{'goal':'读取合成商品后总结'})).data
                        passed=(output['status']=='partial' and output['stop_reason']=='budget_exhausted'
                                and output['candidates']==[] and output['observed_product_count']==2
                                and output['evidence_refs']==['ctx_synthetic'] and output['feedback']==[])
                passed=passed and calls==['search'] and len(usage)==1
                report={'role':role,'passed':passed,'calls':calls,'model_calls':len(usage),'output':output}
            except Exception as error:
                report={'role':role,'passed':False,'error_type':type(error).__name__}
            finally:
                init_budget(0);ShoppingContext.reset(token);context_usage_sink.reset(usage_token)
            report.update(model=settings.llm_model,elapsed_s=round(time.monotonic()-started,3),
                scope='真实模型 + 生产执行边界；测试目录；受控注入预算耗尽；无真实交易')
            reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
        await model.aclose()
    if not all(r['passed'] for r in reports):raise SystemExit(1)


if __name__=='__main__':
    asyncio.run(main())
