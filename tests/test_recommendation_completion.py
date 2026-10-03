"""原生图在推荐成功后结束；失败、普通咨询和下一轮不被误截断。"""
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from app.application.runtime.delivery import FinalDeliveryMiddleware
from app.application.runtime.tools import as_langchain_tool
from app.application.tools.order_tools import _ok, _fail


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(identifier):
    return AIMessage(content="", tool_calls=[{"id": identifier, "name": "recommend_products", "args": {}}])


@pytest.mark.asyncio
async def test_failed_recommendation_can_correct_and_success_stops_then_new_turn_continues():
    attempts = []
    async def recommend_products():
        """测试推荐交付。"""
        attempts.append(1)
        return _fail("超过预算") if len(attempts) == 1 else _ok({"guidance": "通勤更看重轻便时，我推荐这款；需要防雨时再看另一款。", "hits": [{"product_id": "P1001"}]})
    model = Model(responses=[call("first"), call("second"), AIMessage(content="下一轮正常咨询"), AIMessage(content="不应调用")])
    graph = create_agent(model, tools=[as_langchain_tool(recommend_products)],
        middleware=[FinalDeliveryMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test"}}
    result = await graph.ainvoke({"messages": [HumanMessage(content="推荐商品")]}, config)
    assert model.i == 2 and len(attempts) == 2
    assert result["messages"][-1].content == "通勤更看重轻便时，我推荐这款；需要防雨时再看另一款。"
    assert result['product_delivery_complete'] is True
    result = await graph.ainvoke({"messages": [HumanMessage(content="再问一个问题")]}, config)
    assert model.i == 3 and result["messages"][-1].content == "下一轮正常咨询"
    assert result['product_delivery_complete'] is False


@pytest.mark.asyncio
async def test_other_failed_or_unfinished_tools_are_not_hidden():
    middleware = FinalDeliveryMiddleware()
    calls = AIMessage(content="", tool_calls=[{"id": "a", "name": "recommend_products", "args": {}},
        {"id": "b", "name": "query_order_tool", "args": {}}])
    success = ToolMessage(name="recommend_products", tool_call_id="a", content="ok")
    assert await middleware.abefore_model({"messages": [calls, success]}, None) is None
    failure = ToolMessage(name="query_order_tool", tool_call_id="b", content="error", status="error")
    assert await middleware.abefore_model({"messages": [calls, success, failure]}, None) is None


@pytest.mark.asyncio
async def test_ordinary_conversation_unchanged():
    model = Model(responses=[AIMessage(content="你好"), AIMessage(content="不应调用")])
    graph = create_agent(model, tools=[], middleware=[FinalDeliveryMiddleware()])
    result = await graph.ainvoke({"messages": [HumanMessage(content="你好")]})
    assert model.i == 1 and result["messages"][-1].content == "你好"


@pytest.mark.asyncio
async def test_successful_old_delivery_without_guidance_is_not_given_placeholder_advice():
    middleware = FinalDeliveryMiddleware()
    request = call("legacy")
    receipt = ToolMessage(name="recommend_products", tool_call_id="legacy", content="旧交付",
                          artifact={"data": {"hits": [{"product_id": "P1001"}]}})
    with pytest.raises(ValueError, match="整体选购建议"):
        await middleware.abefore_model({"messages": [request, receipt]}, None)
