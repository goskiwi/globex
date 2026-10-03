# -*- coding: utf-8 -*-
"""Product 聚合根

Globex 把跨境商品建模为 Product（SPU）+ Sku（多个），携带品牌、产地、亮点等结构化属性。
SearchAgent 召回的"候选集"传递的就是 Product 卡片，TradeAgent 创建订单时再以 Sku 粒度结算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

from app.domain.catalog.sku import Sku


@dataclass(frozen=True)
class ProductHighlight:
    label: str
    detail: str = ""


@dataclass
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    description: str
    highlights: list[ProductHighlight] = field(default_factory=list)
    ships_to: list[str] = field(default_factory=list)
    skus: list[Sku] = field(default_factory=list)
    # 以下字段来自版本化商品数据集；用于评测跨平台去重、材质/配送约束与数据新鲜度。
    source_platform: str = ""
    external_product_id: str = ""
    canonical_product_id: str = ""
    material_tags: list[str] = field(default_factory=list)
    # 每个销售单位的净重；未知用 None，不以 0 或序号公式冒充事实。
    weight_kg: float | None = None
    dimensions_cm: dict[str, float] = field(default_factory=dict)
    package_dimensions_cm: dict[str, float] = field(default_factory=dict)
    tax_category: str = ""
    rating_summary: dict[str, float | int] | None = None
    updated_at: str = ""
    source_language: str = ""
    source_locale: str = ""
    localized_category: str = ""
    localized_material: str = ""
    data_provenance: str = ""

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("Product.product_id required")
        if not self.skus:
            raise ValueError(f"Product 至少要有一个 Sku：{self.product_id}")
        if self.weight_kg is not None and (not math.isfinite(self.weight_kg) or self.weight_kg <= 0):
            raise ValueError(f"Product.weight_kg 已知时必须为有限正数：{self.product_id}")

    def primary_sku(self) -> Sku:
        return self.skus[0]

    def primary_available_sku(self) -> Sku:
        """优先返回可售 SKU，避免部分缺货商品仍展示缺货的默认规格。"""
        return next((sku for sku in self.skus if sku.stock > 0), self.primary_sku())

    def has_available_sku(self) -> bool:
        return any(sku.stock > 0 for sku in self.skus)

    def find_sku(self, sku_id: str) -> Optional[Sku]:
        return next((s for s in self.skus if s.sku_id == sku_id), None)

    def searchable_text(self, *, skus: list[Sku] | None = None) -> str:
        """商品事实与逐条规格共同参与检索；价库存由业务字段核验。

        建库和粗召回读取全部规格，精排可显式传入本次合格规格。
        每个规格单独成行，不把不同 SKU 的选项拼成不存在的组合。
        """
        highlight_text = " ".join(f"{h.label} {h.detail}" for h in self.highlights)
        facts = " ".join(
            [
                self.title, self.brand, self.localized_category or self.category,
                self.origin_country, self.description, highlight_text,
                self.localized_material or " ".join(self.material_tags),
                self.localized_category or self.tax_category,
            ],
        )
        specifications = sorted({sku.spec.strip() for sku in (self.skus if skus is None else skus) if sku.spec.strip()})
        return "\n".join([facts, *(f"规格：{spec}" for spec in specifications)])
