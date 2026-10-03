from tests.shopping_state_helpers import work_fixture
"""LangChain 消息上的上下文保护与失败回滚；不使用旧 Agent/Toolkit 替身。"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.application.runtime.context import RequestContextMiddleware
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from tests.native_context_helpers import summary_selection
from app.application.runtime.context_summary import render_summary


def history():
    messages = []
    for index in range(5):
        messages.extend([
            HumanMessage(content="找商品", name="buyer", id=f"u{index}"),
            AIMessage(content="", id=f"a{index}", tool_calls=[{
                "id": f"c{index}", "name": "product_search_tool", "args": {},
            }]),
            ToolMessage(id=f"t{index}", name="product_search_tool", tool_call_id=f"c{index}",
                content=json.dumps({"hits": [{"product_id": f"P100{index}",
                    "description": "商品描述" * 100}]}, ensure_ascii=False), artifact={"data":{"hits": [{"product_id": f"P100{index}",
                    "description": "商品描述" * 100}]}}),
        ])
    return messages


def policy(tmp_path, response=None):
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content=summary_selection() if response is None else response)))
    settings = SimpleNamespace(context_size=128000, context_target_tokens=100000,
                               context_product_tokens=1, tool_result_limit=12000)
    return RequestContextMiddleware(ContextEvidenceStore(tmp_path / "evidence.db"), model, settings,
                                    system_prompt="",tools=[])


async def test_unread_and_latest_two_results_are_protected(tmp_path):
    middleware = policy(tmp_path)
    messages = history()
    state = {"messages": messages, "read_tool_messages": ["t1", "t2", "t3", "t4"]}
    token = ShoppingContext.set(ShoppingContextSnapshot("s", "buyer", "zh-CN", "CNY"))
    try:
        update = await middleware.compact_checkpoint(state)
        assert [message.id for message in update["messages"]] == ["t1", "t2"]
        archived = json.loads(update["messages"][0].content)
        assert archived["archived"]
        saved = await middleware.store.get("buyer", "s", archived["result_ref"])
        assert saved["data"] == json.loads(messages[5].content)
        assert all("archived" not in str(message.content) for message in messages)
        middleware.model.ainvoke.assert_not_awaited()
    finally:
        ShoppingContext.reset(token)


async def test_checkpoint_selection_protects_evidence_when_latest_query_omits_sku(tmp_path):
    middleware=policy(tmp_path)
    messages=history()
    state={'messages':messages,'read_tool_messages':['t0','t1','t2','t3','t4'],
           'shopping_work':work_fixture(selected=['P1000'])}
    token=ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'))
    try:
        update=await middleware.compact_checkpoint(state)
        assert [message.id for message in update['messages']]==['t1','t2']
        assert 'archived' not in messages[2].content
    finally:ShoppingContext.reset(token)


async def test_rejected_summary_never_replaces_context(tmp_path):
    middleware = policy(tmp_path, "商品 P99999 已下单")
    state = {"messages": history(), "read_tool_messages": ["t0", "t1", "t2", "t3", "t4"]}
    original = [message.model_dump() for message in state["messages"]]
    token = ShoppingContext.set(ShoppingContextSnapshot("s", "buyer", "zh-CN", "CNY"))
    try:
        with pytest.raises(ValueError, match="来源"):
            await middleware.compact_checkpoint(state, force=True)
        assert [message.model_dump() for message in state["messages"]] == original
        rejected = await middleware.store.search("buyer", "s", kind="rejected_summary", limit=1)
        assert rejected == []  # 拒绝候选不能进入事实回查。
        with middleware.store._connect() as db:
            row = db.execute("SELECT payload FROM context_evidence WHERE kind='rejected_summary'").fetchone()
        assert json.loads(row[0])["candidate"] == "商品 P99999 已下单"
    finally:
        ShoppingContext.reset(token)


@pytest.mark.parametrize('pending',[False,True])
async def test_summary_retains_three_recent_turns_and_unresolved_calls(tmp_path,pending):
    middleware=policy(tmp_path)
    messages=[]
    for index in range(6):
        messages.extend([HumanMessage(name='buyer',content=f'第{index}轮',id=f'u{index}'),
                         AIMessage(content='已记录',id=f'a{index}')])
    if pending:
        messages[1]=AIMessage(content='',id='a0',tool_calls=[{'id':'pending','name':'remember_preference_tool','args':{}}])
    before=[m.model_dump_json() for m in messages]
    token=ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'))
    try:
        update=await middleware.compact_checkpoint({'messages':messages},force=True)
        if pending:
            assert update['messages']==[]
            middleware.model.ainvoke.assert_not_awaited()
        else:
            from langgraph.graph.message import add_messages
            retained=add_messages(messages,update['messages'])
            assert [m.id for m in retained[1:]]==[m.id for m in messages[6:]]
        assert [m.model_dump_json() for m in messages]==before
    finally:ShoppingContext.reset(token)


def test_summary_boundary_does_not_split_cross_turn_tool_pair():
    from app.application.runtime.context import summary_boundary
    messages=[HumanMessage(name='buyer',content='旧需求',id='u0'),
        AIMessage(content='',tool_calls=[{'id':'call','name':'read','args':{}}]),
        HumanMessage(name='buyer',content='补充',id='u1'),
        ToolMessage(tool_call_id='call',content='完成'),
        HumanMessage(name='buyer',content='随后',id='u2'),
        HumanMessage(name='buyer',content='本轮',id='u3')]
    assert summary_boundary(messages,[0,2,4,5],set())==0


async def test_native_compaction_survives_sqlite_reopen_and_next_turn(tmp_path):
    from langchain.agents import create_agent
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from tests.test_langgraph_runtime import ScriptedModel
    middleware=policy(tmp_path)
    config={'configurable':{'thread_id':'s'}}
    messages=[]
    for index in range(6):
        messages.extend([HumanMessage(name='buyer',content=f'第{index}轮',id=f'u{index}'),
                         AIMessage(content='已记录',id=f'a{index}')])
    token=ShoppingContext.set(ShoppingContextSnapshot('s','buyer','zh-CN','CNY'))
    try:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'graph.db')) as saver:
            graph=create_agent(ScriptedModel(),middleware=[middleware],checkpointer=saver)
            await graph.ainvoke({'messages':messages},config)
            before=await graph.aget_state(config)
            update=await middleware.compact_checkpoint(before.values,force=True)
            await graph.aupdate_state(config,update)
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'graph.db')) as saver:
            graph=create_agent(ScriptedModel(),middleware=[middleware],checkpointer=saver)
            restored=await graph.aget_state(config)
            assert '第0轮' in render_summary(restored.values['context_summary'])
            assert [m.id for m in restored.values['messages'][1:]]==[m.id for m in before.values['messages'][6:]]
            result=await graph.ainvoke({'messages':[HumanMessage(name='buyer',content='继续')]},config)
            assert result['messages'][-1].text=='LangGraph 运行正常'
    finally:ShoppingContext.reset(token)
