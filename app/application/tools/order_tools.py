# -*- coding: utf-8 -*-
"""订单工具集：下单准备与完整订单任务

MainAgent 单干与 TradeAgent 派发两条路径共用。写工具只准备确认凭证，不执行用户决议。

函数签名直接用于 LangChain 工具 schema。
"""
import json
from langchain.tools import ToolRuntime
from app.application.agents.shopping_state import Choice
from app.application.tools.shopping_operations import selected_work,work_update

from app.application.runtime.results import ToolResult, ToolResultState

from app.application.usecases.order_usecases import (
    OrderItemInput,
    PlaceOrderUseCase,
)
from app.domain.order.address import Address
from app.application.catalog_evidence import verified_sku_items
from app.application.tools.order_inputs import CreateOrderInput, ShippingAddressInput, CancellationInput, HandleOrderInput
from app.application.runtime.request_scope import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def _ok(payload: dict) -> ToolResult:
    return ToolResult(data=payload, state=ToolResultState.SUCCESS)


def _fail(message: str) -> ToolResult:
    return ToolResult(data=f"[error] {message}", state=ToolResultState.ERROR,
                      error_code="business_rejected", error_reason=message)


def _identity():
    snapshot = ShoppingContext.current()
    if snapshot is None or not snapshot.buyer_id or not snapshot.shopping_session_id:
        raise ValueError("订单操作缺少已绑定的买家与会话身份")
    return snapshot.buyer_id, snapshot.shopping_session_id


async def _verified_order_items(evidence_store, buyer_id: str, session_id: str, items: list[dict]) -> list[OrderItemInput]:
    """从会话原始工具证据核对来源，模型文字及当前目录不能替代检索记录。"""
    if evidence_store is None:
        raise ValueError("商品检索证据未接入，不能准备下单确认")
    return await verified_sku_items(evidence_store.find_product, buyer_id, session_id, items)


def build_create_order_tool(usecase: PlaceOrderUseCase, bus: TradeEventBus, evidence_store=None):
    async def create_order_tool(
        items: list[Choice],
        shipping_address: ShippingAddressInput,
        runtime: ToolRuntime = None,
    ) -> ToolResult:
        """准备下单意向的权威确认卡，返回 confirmation_required，不创建订单或扣库存。

        即使买家在对话中说“同意”，也必须等待其点击页面确认卡；模型不能代为确认。
        金额包含商品、运费和税费，不代表付款。买家身份由系统会话上下文注入。
        商品与规格必须来自当前买家、当前会话的检索结果；否则先精确检索。
        直接提交本次明确选择的订单行；后端校验来源、保存选项并准备确认，无需先调详情、报价或选购工具。
        当前价库存、配送与完整到手报价由确认服务重新核验，不能以模型文字代替。

        """
        try:
            buyer_id, session_id = _identity()
        except ValueError as err:
            return _fail(str(err))
        try:
            request = CreateOrderInput(items=items, shipping_address=shipping_address)
            items = request.items
            bus.publish(session_id, "tool.invoke", {"tool": "create_order_tool", "args": {"buyer_id": buyer_id, "items": [x.model_dump() for x in items]}})
            if runtime is not None:
                from app.application.runtime.handoff import HandoffContext
                if isinstance(runtime.context, HandoffContext):
                    scoped = {line['sku_id']: line for line in ShoppingContext.current().selected_lines}
                    if any(scoped.get(item.sku_id) != item.model_dump() for item in items):
                        raise ValueError('交易子任务不能更换父任务绑定的SKU或数量')
            shipping_address = request.shipping_address.model_dump()
            order_items = await _verified_order_items(evidence_store, buyer_id, session_id, [x.model_dump() for x in items])
            work = selected_work(runtime, items, []) if runtime is not None and 'shopping_work' in runtime.state else None
            address = Address(
                recipient_name=shipping_address.get("recipient_name", ""),
                country=shipping_address.get("country", ""),
                state=shipping_address.get("state", ""),
                city=shipping_address.get("city", ""),
                address_line=shipping_address.get("address_line", ""),
                postal_code=shipping_address.get("postal_code", ""),
                phone=shipping_address.get("phone", ""),
            )
            result = await usecase.execute(
                buyer_id=buyer_id, session_id=session_id, items=order_items, shipping_address=address, currency=(ShoppingContext.current().effective_search or {}).get("parameters", {}).get("target_currency") or ShoppingContext.current().currency,
            )
        except (ValueError, KeyError, TypeError) as err:
            return _fail(str(err))
        return ToolResult(result, state_updates=work_update(runtime,work) if work is not None else None)

    create_order_tool.input_model = CreateOrderInput
    return create_order_tool


def build_handle_order_tool(usecase, bus: TradeEventBus):
    async def handle_order(order_id: str, cancellation: CancellationInput | None = None) -> ToolResult:
        """查看当前买家的完整订单；携带cancellation时在同一任务中准备取消确认卡。

        返回订单查看快照、操作结果与可选确认单。只查看不准备或取消订单；准备取消不回补库存，
        用户点击页面确认卡后才执行。订单归属、当前状态和账单由服务端事务核验。
        """
        try:
            buyer_id, session_id = _identity()
            request = HandleOrderInput(order_id=order_id, cancellation=cancellation)
            bus.publish(session_id, 'tool.invoke', {'tool':'handle_order','args':request.model_dump()})
            result = await usecase.execute(request.order_id, buyer_id=buyer_id, session_id=session_id,
                cancellation=request.cancellation.model_dump() if request.cancellation is not None else None)
        except ValueError as error:
            return _fail(str(error))
        return _ok(result)
    return handle_order
