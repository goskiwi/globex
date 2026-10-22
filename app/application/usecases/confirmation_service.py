# -*- coding: utf-8 -*-
"""权威交易确认：工具准备快照，用户 HTTP 决议才可进入原子交易底座。"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.order.address import Address
from app.application.usecases.order_usecases import OrderItemInput
from app.application.usecases.pricing import PricingService, QuoteItem

if TYPE_CHECKING:
    from app.domain.order.ports.trade_store import TradeStore
    from app.infrastructure.eventbus import TradeEventBus

logger = logging.getLogger(__name__)


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空")
    return value


class ConfirmationService:
    def __init__(
        self, product_repo: ProductRepository, trade_store: TradeStore, bus: TradeEventBus | None = None,
        ttl_seconds: int = 300, *, pricing=None, clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError("确认有效期必须为正整数秒")
        self._product_repo = product_repo
        self._pricing = pricing or PricingService(product_repo)
        self._trade_store = trade_store
        self._bus = bus
        self._ttl = timedelta(seconds=ttl_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _identity(buyer_id: str, session_id: str) -> None:
        _required_text(buyer_id, "买家身份")
        _required_text(session_id, "会话标识")

    @staticmethod
    def _envelope(confirmation: dict) -> dict:
        result = {"confirmation_required": confirmation["status"] == "pending", "confirmation": confirmation}
        if confirmation.get("result") is not None:
            result["order"] = confirmation["result"]
        return result

    def _publish(self, session_id: str, event_type: str, confirmation: dict) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(session_id, event_type, {"confirmation": confirmation})
        except Exception as err:  # noqa: BLE001
            # 记录已持久化；事件失败不能把已完成事务误报成失败，HTTP/刷新仍可恢复。
            logger.warning("确认事件投递失败（可从持久存储恢复）：%s", err)

    async def prepare_order(
        self, buyer_id: str, session_id: str, items: list[OrderItemInput], shipping_address: Address, currency: str = "CNY",
    ) -> dict:
        self._identity(buyer_id, session_id)
        if not isinstance(items, list) or not items:
            raise ValueError("PlaceOrder.items 不能为空")
        if not isinstance(shipping_address, Address):
            raise ValueError("收货地址格式无效")
        address = asdict(shipping_address)
        for key, value in address.items():
            if not isinstance(value, str):
                raise ValueError(f"收货地址字段 {key} 必须是文字")
            address[key] = value.strip()
        for key in ("recipient_name", "country", "city", "address_line"):
            _required_text(address[key], f"收货地址 {key}")

        quantities: dict[tuple[str, str], int] = {}
        for item in items:
            if not isinstance(item, OrderItemInput):
                raise ValueError("订单明细必须为 OrderItemInput")
            if type(item.quantity) is not int or item.quantity <= 0:
                raise ValueError("订单数量必须为正整数")
            key = (_required_text(item.product_id, "商品标识"), _required_text(item.sku_id, "规格标识"))
            quantities[key] = quantities.get(key, 0) + item.quantity

        products = {p.product_id: p for p in await self._product_repo.find_by_ids(
            list(dict.fromkeys(p for p, _ in quantities)))}
        pricing = self._pricing.quote_observed(
            [QuoteItem(p, s, q) for (p, s), q in quantities.items()], products, address["country"], currency)
        confirmation = await self._trade_store.prepare_confirmation(
            operation_id=f"operation-{uuid.uuid4().hex}", buyer_id=buyer_id, session_id=session_id,
            action="create", payload={**pricing, "shipping_address": address},
            expires_at=self._clock() + self._ttl,
        )
        self._publish(session_id, "confirmation.required", confirmation)
        return self._envelope(confirmation)

    async def prepare_cancel(self, buyer_id: str, session_id: str, order_id: str, reason: str) -> dict:
        self._identity(buyer_id, session_id)
        _required_text(order_id, "订单号")
        _required_text(reason, "取消原因")
        confirmation = await self._trade_store.prepare_confirmation(
            operation_id=f"operation-{uuid.uuid4().hex}", buyer_id=buyer_id, session_id=session_id,
            action="cancel", payload={"order_id": order_id, "reason": reason.strip(),
                "amount_scope": "landed", "order_kind": "purchase_intent"},
            expires_at=self._clock() + self._ttl,
        )
        self._publish(session_id, "confirmation.required", confirmation)
        return self._envelope(confirmation)

    async def handle_order(self, buyer_id: str, session_id: str, order_id: str, cancellation: dict | None = None) -> dict:
        self._identity(buyer_id, session_id)
        _required_text(order_id, '订单号')
        request = None
        if cancellation is not None:
            request = {'reason': _required_text(cancellation['reason'], '取消原因').strip(),
                'operation_id': 'operation-' + uuid.uuid4().hex, 'expires_at': self._clock() + self._ttl}
        result = await self._trade_store.handle_order(order_id, buyer_id=buyer_id, session_id=session_id, cancellation=request)
        confirmation = result['confirmation']
        if confirmation is not None:
            self._publish(session_id, 'confirmation.required', confirmation)
        rejection = result['rejection']
        return {'order':result['order'], 'operation': {'kind':'prepare_cancel' if request is not None else 'inspect',
                'status':'rejected' if rejection else 'awaiting_confirmation' if confirmation is not None else 'completed',
                'rejection': rejection},
            'confirmation_required': confirmation is not None and confirmation['status']=='pending',
            'confirmation': confirmation, 'status':'partial' if rejection else 'completed'}

    async def resolve(
        self, confirmation_id: str, buyer_id: str, session_id: str, snapshot_hash: str, approved: bool,
    ) -> dict:
        """仅供用户动作入口调用；不注册为模型工具。底座原子校验、幂等与扣补库存。"""
        self._identity(buyer_id, session_id)
        _required_text(confirmation_id, "确认凭证")
        _required_text(snapshot_hash, "快照校验值")
        if type(approved) is not bool:
            raise ValueError("确认决定必须为布尔值，不能使用模型文字或字符串")
        confirmation = await self._trade_store.resolve_confirmation(
            confirmation_id, buyer_id=buyer_id, session_id=session_id,
            snapshot_hash=snapshot_hash, approved=approved,
        )
        self._publish(session_id, "confirmation.resolved", confirmation)
        return self._envelope(confirmation)

    async def get(self, confirmation_id: str, buyer_id: str, session_id: str) -> dict:
        self._identity(buyer_id, session_id)
        _required_text(confirmation_id, "确认凭证")
        confirmation = await self._trade_store.get_confirmation(
            confirmation_id, buyer_id=buyer_id, session_id=session_id,
        )
        return self._envelope(confirmation)

    async def list(self, buyer_id: str, session_id: str, limit: int = 20) -> dict:
        self._identity(buyer_id, session_id)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("确认列表数量必须为 1 至 20 的整数")
        return {"confirmations": await self._trade_store.list_confirmations(
            buyer_id=buyer_id, session_id=session_id, limit=limit,
        )}

    async def agent_state(self, buyer_id: str, session_id: str) -> dict:
        """向模型提供精简的权威交易状态，不重复注入收货地址或确认校验值。"""
        self._identity(buyer_id, session_id)
        snapshot = await self._trade_store.session_trade_state(buyer_id=buyer_id, session_id=session_id)
        return {
            "pending_confirmations": [{
                "action": item["action"], "expired": item["expired"],
                "items": [{"product_id": line["product_id"], "sku_id": line["sku_id"], "quantity": line["quantity"]}
                          for line in item["payload"]["items"]],
                "order_id": item["payload"].get("order_id"),
            } for item in snapshot["pending_confirmations"]],
            "orders": [{key: order.get(key) for key in (
                "order_id", "status", "currency", "total_amount_major", "lines", "cancel_reason",
            )} for order in snapshot["orders"]],
        }

    async def pending(self, buyer_id: str, session_id: str) -> list[dict]:
        """业务决策读取有效操作；list 仅提供最近记录展示。"""
        self._identity(buyer_id, session_id)
        return await self._trade_store.pending_confirmations(buyer_id=buyer_id, session_id=session_id)

    async def restore(self, buyer_id: str, session_id: str) -> dict:
        self._identity(buyer_id, session_id)
        return {"confirmations": await self._trade_store.restore_confirmations(buyer_id=buyer_id, session_id=session_id)}
