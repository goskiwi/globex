"""真实模型自主选择 Skill；测试目录与临时资料库，不接真实买家或交易。"""
import asyncio
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from contextvars import ContextVar
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.agents.main_agent import MainAgentFactory
from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.agents.search_agent import SearchAgentFactory
from app.application.agents.trade_agent import TradeAgentFactory
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.harness.loop_detector import LoopDetector
from app.infrastructure.buyer_skills import BuyerSkillStore
from app.infrastructure.capability_registry import CapabilityRegistry
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_usage import evaluation_evidence_sink
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import create_chat_model, create_chat_client
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore
from app.infrastructure.resilience import CircuitBreakerRegistry
from app.infrastructure.settings import load_settings
from app.infrastructure.throttle import GatewayThrottle


async def main():
    reports=[]
    with tempfile.TemporaryDirectory(prefix='globex-skill-live-') as folder:
        settings=replace(load_settings(),data_dir=Path(folder),llm_fallback_model='',llm_max_retries=0,harness_enabled=True)
        registry=CapabilityRegistry(Path(folder)/'skills.db')
        personal=BuyerSkillStore(Path(folder)/'personal.db')
        skill=personal.save('synthetic','通勤背包比较','需要从用途、重量、预算出发，比较多款通勤背包并说明取舍时使用。',
            '核对当前预算与目的地。用商品搜索工具获取至少两款候选，按目录重量和价格比较；缺少字段如实说明，不猜数值。')
        unrelated=personal.save('synthetic','耳机续航研究','耳机续航和降噪比较时使用。','SYNTHETIC_UNRELATED_BODY：只讨论耳机。')
        model=create_chat_model(settings,stream=False, client=create_chat_client(settings))
        model_client=model.client
        try:
            for scenario in ('simple','research','child'):
                started=time.monotonic();samples=[]
                session='skill-live-'+scenario
                token=ShoppingContext.set(ShoppingContextSnapshot(session,'synthetic','zh-CN','CNY',
                    capability_digest=registry.bind_session(session,'synthetic')))
                sink=evaluation_evidence_sink.set(samples.append)
                bus=TradeEventBus();circuits=CircuitBreakerRegistry();throttle=GatewayThrottle(3,0)
                search=SearchAgentFactory(settings,CatalogSearchUseCase(InMemoryProductRepository()),bus,None,circuits,throttle, model_client=model_client)
                trade=TradeAgentFactory(settings,None,None,None,bus,circuits,throttle, model_client=model_client)
                factory=MainAgentFactory(settings,search,trade,bus,JsonFilePreferenceStore(Path(folder)/'preferences'),
                    circuits,throttle,checkpointer=InMemorySaver(),loop_detector=LoopDetector(),
                    capability_registry=registry,buyer_skill_store=personal, model_client=model_client)
                try:
                    with (
                        patch('app.application.agents.main_agent.create_chat_model',return_value=model),
                        patch('app.application.agents.search_agent.create_chat_model',return_value=model),
                        patch('app.application.agents.trade_agent.create_chat_model',return_value=model),
                    ):
                        async with asyncio.timeout(90):
                            if scenario=='child':
                                dispatch=build_task_dispatch_tool(search,trade,bus)
                                output=(await dispatch('search_agent',{'goal':'比较300元以内寄中国的轻便通勤背包',
                                    'filters':{'landed_budget_major':300,'target_currency':'CNY','ship_to':'CN'},
                                    'skill_refs':[{'id':skill['id'],'version':'1'}]})).data
                            else:
                                query='查一下 P1003-S1 的价格和库存。' if scenario=='simple' else (
                                    '预算300元，寄中国，想找轻便通勤背包。请从选购标准开始，找至少两款比较重量和价格，说明各自取舍。')
                                graph=factory.build().graph
                                runner=MainAgentOrchestrator.__new__(MainAgentOrchestrator)
                                runner._bus=bus;runner._native_observer=ContextVar('skill-live-observer',default=None)
                                graph_session=SimpleNamespace(graph=graph,config={'configurable':{'thread_id':session}})
                                answer=await runner._reply(session,graph_session,[HumanMessage(name='synthetic',content=query)])
                                result=(await graph.aget_state(graph_session.config)).values
                                output={'answer':answer.text,'loaded_skills':result.get('loaded_skills',{}),
                                        'stop_reason':result['messages'][-1].additional_kwargs.get('execution_stop')}
                    requests=[s['payload'] for s in samples if s['kind']=='model_request']
                    loads=[s['payload'] for s in samples if s['kind']=='tool_result'
                           and s['payload'].get('tool')=='load_agent_skill_tool' and s['payload'].get('state')=='success']
                    first=json.dumps(requests[0].get('messages',[]),ensure_ascii=False) if requests else ''
                    all_messages=json.dumps([r.get('messages',[]) for r in requests],ensure_ascii=False)
                    body=skill['body']
                    passed=bool(requests) and body not in first and 'SYNTHETIC_UNRELATED_BODY' not in all_messages
                    passed=passed and (not loads if scenario=='simple' else bool(loads) and body in all_messages)
                    if scenario=='child':passed=passed and output['status'] in ('completed','partial')
                    report={'scenario':scenario,'passed':passed,'assessment':'仅评估渐进加载；任务完成状态单独列出',
                            'task_status':output.get('status') if scenario=='child' else ('partial' if output['stop_reason'] else 'answered'),
                            'model_requests':len(requests),
                            'skill_loads':len(loads),'output':output}
                except Exception as error:
                    import traceback
                    report={'scenario':scenario,'passed':False,'error_type':type(error).__name__,
                        'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                                  for f in traceback.extract_tb(error.__traceback__)[-5:]]}
                finally:
                    evaluation_evidence_sink.reset(sink);ShoppingContext.reset(token)
                report.update(model=settings.llm_model,elapsed_s=round(time.monotonic()-started,3),
                    tool_trace=[{'tool':s['payload'].get('tool'),'state':s['payload'].get('state'),
                                 'result':s['payload'].get('result')} for s in samples if s['kind']=='tool_result'],
                    scope='真实模型；生产工厂和加载链；测试目录及合成 Skill；无真实交易')
                print(json.dumps(report,ensure_ascii=False),flush=True);reports.append(report)
        finally:await model.aclose()
    if not all(r['passed'] for r in reports):raise SystemExit(1)


if __name__=='__main__':
    asyncio.run(main())
