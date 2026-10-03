"""统一 SKU 计价：同 SKU 合并数量，各 SKU 独立计运税，再以目标币种汇总。"""
from dataclasses import dataclass

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
from app.domain.shipping.tariff_schedule import TariffSchedule


@dataclass(frozen=True)
class QuoteItem:
    product_id: str
    sku_id: str
    quantity: int

    def __post_init__(self):
        if not self.product_id or not self.sku_id or type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError("报价必须提供商品、规格及正整数数量")


class PricingService:
    def __init__(self, product_repo, tariff=None):
        self.product_repo = product_repo
        self.tariff = tariff or TariffSchedule(ExchangeRateTable())

    def price_sku(self, product, sku, quantity, ship_to, currency):
        QuoteItem(product.product_id, sku.sku_id, quantity)
        if ship_to not in product.ships_to:
            raise ValueError(f"商品不支持配送至 {ship_to}：{product.product_id}")
        unit = self.tariff.rates.convert(sku.price, currency)
        shipping = self.tariff.quote(unit.multiply(quantity), product.category, ship_to, quantity, currency)
        return {
            "product_id": product.product_id, "sku_id": sku.sku_id,
            "title": f"{product.title}（{sku.spec}）", "quantity": quantity,
            "unit_price_minor": unit.amount_in_minor_units, "currency": currency,
            "source_unit_price_minor": sku.price.amount_in_minor_units, "source_currency": sku.price.currency,
            "subtotal_minor": shipping.subtotal.amount_in_minor_units,
            "freight_minor": shipping.freight.amount_in_minor_units,
            "tariff_minor": shipping.tariff.amount_in_minor_units,
            "total_amount_minor": shipping.landed_total().amount_in_minor_units,
        }

    @staticmethod
    def assemble(lines, ship_to, currency):
        if not lines:
            raise ValueError("报价至少需要一件商品")
        return {"items": lines, "ship_to": ship_to, "currency": currency,
                **{key: sum(line[key] for line in lines) for key in
                   ("subtotal_minor", "freight_minor", "tariff_minor", "total_amount_minor")}}

    async def quote(self, items: list[QuoteItem], ship_to: str, currency: str):
        quantities = {}
        for item in items:
            key = (item.product_id, item.sku_id)
            quantities[key] = quantities.get(key, 0) + item.quantity
        lines = []
        for (product_id, sku_id), quantity in sorted(quantities.items()):
            product = await self.product_repo.find_by_id(product_id)
            sku = product.find_sku(sku_id) if product else None
            if sku is None:
                raise ValueError(f"商品或规格不存在：{product_id}/{sku_id}")
            if sku.stock < quantity:
                raise ValueError(f"库存不足：{sku_id}")
            lines.append(self.price_sku(product, sku, quantity, ship_to, currency))
        return self.assemble(lines, ship_to, currency)
