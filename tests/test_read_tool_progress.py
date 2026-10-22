"""进展观察取工具只读声明，不靠工具名称名单。"""
import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage,AIMessage
from app.application.harness.loop_detector import LoopDetector
from app.application.runtime.execution import ExecutionMiddleware
from app.application.runtime.loops import LoopMiddleware
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.results import ToolResult
from app.application.runtime.tools import as_langchain_tool
from app.infrastructure.eventbus import TradeEventBus
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_langgraph_runtime import _container


@pytest.mark.parametrize('name',['handle_order','quote_products','future_read_operation'])
def test_every_declared_read_has_repeat_detection_and_changed_values_are_progress(name):
    detector=LoopDetector();state={}
    for _ in range(4):
        observation=detector.observation(name,{}, {'status':'unchanged'},'success',{},read_only=True)
        state,_=detector.advance(state,[observation])
    assert state['stop']
    fresh={}
    for i in range(6):
        fresh,hint=detector.advance(fresh,[detector.observation(name,{}, {'stock':i},'success',{},read_only=True)])
        assert hint is None and not fresh['stop']
    assert detector.observation(name,{}, {},'success',{},read_only=False) is None


async def test_native_tool_metadata_drives_loop_feedback():
    executed=[]
    async def future_read_operation():
        """返回同一合成事实。"""
        executed.append(1);return ToolResult({'status':'unchanged'})
    detector=LoopDetector();model=ScriptedModel(responses=[call('future_read_operation',{}) for _ in range(4)]+[AIMessage(content='已有结论')])
    graph=create_agent(model,tools=[as_langchain_tool(future_read_operation,read_only=True)],middleware=[BusinessToolMiddleware(detector,TradeEventBus()),LoopMiddleware(detector),ExecutionMiddleware(10,main=True)])
    state=await graph.ainvoke({'messages':[HumanMessage(content='查询')]})
    assert len(executed)==4 and state['execution_stop']=='repeated_path'
    assert any(m.name=='loop_feedback' for m in state['messages'])


async def test_actual_main_tools_declare_reads_and_keep_writes_out(tmp_path,monkeypatch):
    c=await _container(tmp_path,monkeypatch,ScriptedModel())
    try:
        tools={t.name:t for t in c.orchestrator._sessions._main_factory.build().context_policy.tools}
        for name in ['product_search_tool','category_insight_tool','conversation_fact_lookup','quote_products','load_agent_skill_tool','lookup_strategy_memory_tool']:
            assert tools[name].metadata['read_only']
        for name in ['create_order_tool','handle_order','recommend_products','remember_preference_tool','update_preference_tool','forget_preference_tool','task_dispatch']:
            assert not tools[name].metadata['read_only']
    finally:await c.shutdown()
