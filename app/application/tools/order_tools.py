# -*- coding: utf-8 -*-
"""订单工具集：create_order_tool / query_order_tool / cancel_order_tool

MainAgent 单干与 TradeAgent 派发两条路径共用。写工具只准备确认凭证，不执行用户决议。

函数签名直接用于 LangChain 工具 schema。
"""
import json

from app.application.runtime.results import ToolResult, ToolResultState

from app.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from app.domain.order.address import Address
from app.application.tools.order_inputs import CreateOrderInput, ShippingAddressInput
from app.infrastructure.context import ShoppingContext
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
    verified = []
    for item in items:
        product_id, sku_id = item.get("product_id"), item.get("sku_id")
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("订单行必须提供 product_id")
        if not isinstance(sku_id, str) or not sku_id.strip():
            raise ValueError("sku_id 必须为非空文字")
        evidence = await evidence_store.find_product(buyer_id, session_id, product_id=product_id, sku_id=sku_id or "")
        if evidence is None:
            raise ValueError(f"当前会话未检索返回商品或规格：{sku_id or product_id}；请先用 product_search_tool 精确核验")
        card = next((hit for hit in evidence["data"].get("hits", []) if hit.get("product_id") == product_id), None)
        if card is None:
            raise ValueError("商品检索证据与请求不匹配")
        sku_ids = {sku["sku_id"] for sku in card.get("skus", []) if sku.get("sku_id")}
        if sku_id not in sku_ids:
            raise ValueError(f"商品 {product_id} 缺少可核对的规格，请先检索并明确 sku_id")
        verified.append(OrderItemInput(product_id, sku_id, item["quantity"]))
    return verified


def build_create_order_tool(usecase: PlaceOrderUseCase, bus: TradeEventBus, evidence_store=None):
    async def create_order_tool(
        sku_ids: list[str],
        shipping_address: ShippingAddressInput,
    ) -> ToolResult:
        """准备下单意向的权威确认卡，返回 confirmation_required，不创建订单或扣库存。

        即使买家在对话中说“同意”，也必须等待其点击页面确认卡；模型不能代为确认。
        金额包含商品、运费和税费，不代表付款。买家身份由系统会话上下文注入。
        商品与规格必须来自当前买家、当前会话的检索结果；否则先精确检索。
        只接受当前已选 SKU；商品和数量由服务端读取选购记录，不接受模型重填。

        Args:
            sku_ids (`list[str]`):
                本次准备确认的已选 SKU ID，必须先由主代理 update_shopping_state 保存选购项。
            shipping_address (`dict`):
                收货地址，形如 {"recipient_name": "...", "country": "CN", "state": "...",
                "city": "...", "address_line": "...", "postal_code": "...", "phone": "..."}。
        """
        try:
            buyer_id, session_id = _identity()
        except ValueError as err:
            return _fail(str(err))
        bus.publish(session_id, "tool.invoke", {"tool": "create_order_tool", "args": {"buyer_id": buyer_id, "sku_ids": sku_ids}})
        try:
            request = CreateOrderInput(sku_ids=sku_ids, shipping_address=shipping_address)
            sku_ids = request.sku_ids
            shipping_address = request.shipping_address.model_dump()
            selected = {line["sku_id"]: line for line in ShoppingContext.current().selected_lines}
            if any(sku not in selected for sku in sku_ids):
                raise ValueError("SKU 尚未选购，请主代理先用 update_shopping_state 明确规格与数量")
            items = [selected[sku] for sku in sku_ids]
            order_items = await _verified_order_items(evidence_store, buyer_id, session_id, items)
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
        return _ok(result)

    create_order_tool.input_model = CreateOrderInput
    return create_order_tool


def build_query_order_tool(usecase: QueryOrderUseCase, bus: TradeEventBus):
    async def query_order_tool(order_id: str) -> ToolResult:
        """查询当前买家的订单详情；订单号本身不构成读取权限。

        Args:
            order_id (`str`):
                订单号，如 "GBX-000001"。
        """
        try:
            buyer_id, session_id = _identity()
        except ValueError as err:
            return _fail(str(err))
        bus.publish(session_id, "tool.invoke", {"tool": "query_order_tool", "args": {"order_id": order_id}})
        try:
            snapshot = await usecase.execute(order_id, buyer_id=buyer_id)
        except ValueError as err:
            return _fail(str(err))
        return _ok(snapshot)

    return query_order_tool


def build_cancel_order_tool(usecase: CancelOrderUseCase, bus: TradeEventBus):
    async def cancel_order_tool(order_id: str, reason: str) -> ToolResult:
        """准备当前买家的订单取消确认卡，不立即取消或回补库存。

        用户必须点击页面确认卡才能执行。自然语言同意不能代替该用户动作。

        Args:
            order_id (`str`):
                订单号，如 "GBX-000001"。
            reason (`str`):
                取消原因，必填。
        """
        try:
            buyer_id, session_id = _identity()
        except ValueError as err:
            return _fail(str(err))
        bus.publish(session_id, "tool.invoke", {"tool": "cancel_order_tool", "args": {"order_id": order_id, "reason": reason}})
        try:
            result = await usecase.execute(order_id, reason, buyer_id=buyer_id, session_id=session_id)
        except ValueError as err:
            return _fail(str(err))
        return _ok(result)

    return cancel_order_tool
