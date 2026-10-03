"""查询和历史证据只有显式工具路径；措辞不改变工具权限或参数。"""
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool
from app.application.runtime.context import RequestContextMiddleware
from app.application.tools.conversation_fact_lookup import build_conversation_fact_lookup
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from tests.test_retrieval import _settings


@pytest.fixture(autouse=True)
def scope():
    token=ShoppingContext.set(ShoppingContextSnapshot("s","b","zh-CN","CNY"))
    yield
    ShoppingContext.reset(token)


@pytest.mark.parametrize("query",[
    "再查 P1003-S1 的当前价格", "核实 P1003-S1 现在的售价", "确认一下 P1003-S1 的现价",
    "请列出第一批历史SKU的价格", "请列出第一批历史SKU的价格，谢谢", "不要看历史，只查 P1003-S1 当前库存",
])
async def test_request_preserves_tools_model_and_message_roles(tmp_path,query):
    store=ContextEvidenceStore(tmp_path/"e.db")
    payload={"hits":[{"product_id":"P1003","skus":[{"sku_id":"P1003-S1","spec":"黑色","price_major":129,"currency":"CNY","stock":80}]}],"observed_at":"2026-09-30"}
    payload["result_ref"]=await store.save("b","s","products",payload)
    await store.save("b","s","display_batch",payload)
    messages=[HumanMessage(id="u0",name="b",content="查一下商品"),
        AIMessage(id="a0",content="",tool_calls=[{"id":"c0","name":"product_search_tool","args":{}}]),
        ToolMessage(id="t0",tool_call_id="c0",name="product_search_tool",content=json.dumps(payload), artifact={"data":payload}),
        HumanMessage(id="u1",name="b",content=query)]
    original=deepcopy(messages)
    model=SimpleNamespace(streaming=True)
    tool=StructuredTool.from_function(lambda sku_id:sku_id,name="product_search_tool",description="读取当前商品")
    policy=RequestContextMiddleware(store,None,_settings(tmp_path),system_prompt="规则",tools=[tool],summary_enabled=False)
    request=ModelRequest(model=model,messages=messages,system_message=SystemMessage(content="规则"),
        tools=[tool],state={"messages":messages},runtime=None)
    prepared,_,_=await policy.prepare(request)
    assert prepared.model is model and prepared.model.streaming
    assert prepared.tools == [tool] and prepared.tool_choice == request.tool_choice
    assert messages == original
    assert [m.model_dump(exclude={'artifact'}) for m in prepared.messages] == [m.model_dump(exclude={'artifact'}) for m in original]
    assert prepared.messages[2].artifact is None and original[2].artifact is not None
    assert isinstance(prepared.messages[2],ToolMessage)


async def test_explicit_history_selection_paging_and_ownership(tmp_path):
    store=ContextEvidenceStore(tmp_path/"e.db")
    refs=[]
    for price in (10,20):
        refs.append(await store.save("b","s","display_batch",{"observed_at":"2026-09-30","hits":[
            {"product_id":f"P100{i}","price_major":price,"currency":"CNY"} for i in range(7)]}))
    lookup=build_conversation_fact_lookup(store)
    assert not (await lookup()).ok
    first=json.loads((await lookup(batch=1,limit=2)).text)
    assert first["records"][0]["data"]["hits"][0]["price_major"] == 10
    assert first["records"][0]["data"]["next_offset"] == 2
    assert first["records"][0]["observation_scope"]["time_basis"] == "historical"
    latest=json.loads((await lookup(batch=0,product_id="P1000")).text)
    assert latest["records"][0]["data"]["hits"][0]["price_major"] == 20
    by_ref=json.loads((await lookup(result_ref=refs[0],offset=2,limit=2)).text)
    assert by_ref["records"][0]["data"]["hits"][0]["product_id"] == "P1002"
    assert json.loads((await lookup(batch=99)).text)["records"] == []
    token=ShoppingContext.set(ShoppingContextSnapshot("s","other","zh-CN","CNY"))
    try:
        assert json.loads((await lookup(result_ref=refs[0])).text)["records"] == []
    finally: ShoppingContext.reset(token)


async def test_history_word_does_not_disable_pressure_pruning(tmp_path):
    from tests.native_context_helpers import history,policy
    outcomes=[]
    for query in ("继续查询", "历史", "不要历史，只核对库存"):
        state=history()
        state["messages"][-3].content=query
        original=deepcopy(state["messages"])
        updates=await policy(tmp_path,product_tokens=1).compact_checkpoint(state)
        outcomes.append([m.id for m in updates["messages"]])
        assert state["messages"] == original
    assert len(outcomes[0]) == 6 and outcomes[0] == outcomes[1] == outcomes[2]


async def test_latest_display_does_not_fall_back_to_unshown_search(tmp_path):
    store=ContextEvidenceStore(tmp_path/"e.db")
    await store.save("b","s","products",{"hits":[{"product_id":"P1003"}]})
    result=await build_conversation_fact_lookup(store)(batch=0)
    assert json.loads((result).text)["records"] == []


async def test_production_main_sends_auto_tool_choice(tmp_path,monkeypatch):
    import httpx
    from tests.native_model_helpers import client_model,completion
    from tests.test_langgraph_runtime import _container
    from app.application.agents.orchestrator import SubmitIntentInput
    requests=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        assert body.get("tool_choice") in (None,"auto")
        assert body.get("tools")
        return httpx.Response(200,json=completion())
    model=await client_model(tmp_path,handler)
    container=await _container(tmp_path,monkeypatch,model)
    try:
        result=await container.orchestrator.handle_intent(SubmitIntentInput("s","b","zh-CN","CNY","再查 P1003-S1 当前价格"))
        assert not result.error and requests
    finally:
        await container.shutdown()
        await model.aclose()
