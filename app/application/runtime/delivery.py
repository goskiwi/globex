"""最终展示交付：产品与建议一次完成，澄清成功后等待输入，不再调用模型收口。"""
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from typing_extensions import NotRequired
from app.application.usecases.intents import Delivery, DELIVERY_KINDS
from langchain_core.messages import AIMessage, ToolMessage
from app.application.runtime.results import message_data


class DeliveryState(AgentState):
    delivery: NotRequired[Delivery | None]


class FinalDeliveryMiddleware(AgentMiddleware):
    state_schema = DeliveryState

    async def abefore_agent(self, state, runtime):
        return {"delivery": None}
    async def awrap_tool_call(self, request, handler):
        names = DELIVERY_KINDS
        if request.tool_call["name"] in names:
            last = next((m for m in reversed(request.state["messages"]) if isinstance(m, AIMessage)), None)
            if last and sum(c["name"] in names for c in last.tool_calls) > 1:
                return ToolMessage(name=request.tool_call["name"], tool_call_id=request.tool_call["id"],
                    status="error", content="一次只交付一份最终展示，请选择详情、推荐、比较或澄清后单独提交；本批展示未发布。")
        return await handler(request)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        # 只处理刚完成的一批工具；新用户消息不会被历史推荐提前结束。
        messages = state["messages"]
        receipts = []
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break
            receipts.append(message)
        if not receipts or any(m.status != "success" for m in receipts):
            return None
        call = messages[len(messages) - len(receipts) - 1] if len(messages) > len(receipts) else None
        if not isinstance(call, AIMessage) or {c["id"] for c in call.tool_calls} != {m.tool_call_id for m in receipts}:
            return None
        for receipt in receipts:
            if receipt.name == 'present_research':
                payload = message_data(receipt)
                if not isinstance(payload, dict) or not payload.get('guidance') or not payload.get('task_ids'):
                    raise ValueError('研究交付缺少报告或任务来源')
                return {'messages': [AIMessage(content=payload['guidance'])],
                        'delivery': {'kind':'research', 'status':payload['status']}, 'jump_to': 'end'}
            if receipt.name == "show_shopping_form":
                payload = message_data(receipt)
                if not isinstance(payload, dict) or payload.get("status") != "awaiting_input" or not payload.get("form_id"):
                    raise ValueError("澄清交付缺少已保存的表单")
                return {"messages": [AIMessage(content="请填写下方的问题，提交后我会继续选购。")],
                        "delivery": {"kind":"clarification", "status":"needs_input"}, "jump_to": "end"}
            if receipt.name not in {"compare_products", "recommend_products", "show_product_details"}:
                continue
            payload = message_data(receipt)
            if not isinstance(payload, dict) or not isinstance(payload.get("guidance"), str) or not payload["guidance"].strip():
                raise ValueError("最终商品交付缺少整体选购建议")
            return {"messages": [AIMessage(content=payload["guidance"])],
                    "delivery": {"kind": DELIVERY_KINDS[receipt.name],
                                 "status": payload.get("plan_outcome", {}).get("status", "completed")}, "jump_to": "end"}
        return None
