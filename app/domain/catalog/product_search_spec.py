# -*- coding: utf-8 -*-
"""ProductSearchSpec 值对象

SearchAgent 把买家自然语言 query 改写为标准化检索规格：
normalized_query 用于召回，槽位（category / price_band / ship_to / locale）用于过滤。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from app.domain.catalog.taxonomy import Category, CATEGORIES, MATERIAL_TAGS
from typing import Optional
import math
import re


@dataclass(frozen=True)
class ProductSearchSpec:
    normalized_query: str = ""
    category: Category | None = None
    ship_to: Optional[str] = None
    locale: str = "zh-CN"
    top_k: int = 5
    # 到手价目标币种：命中 ship_to 时商品卡内联 landed_price（小计+运费+关税）
    target_currency: str = "CNY"
    # 价格硬约束（目标币种主单位）：硬约束由检索链路结构化过滤，不交给 embedding/reranker
    price_max_major: Optional[float] = None
    # 材质黑名单仅接受明确目录标签，不从自然语言扩大材质范围。
    excluded_material_tags: list[str] | tuple[str, ...] = ()
    # 材质白名单：复合约束评测及“必须是金属/天然纤维”等场景必须结构化过滤。
    required_material_tags: list[str] | tuple[str, ...] = ()
    product_id: Optional[str] = None
    sku_id: Optional[str] = None
    excluded_product_ids: tuple[str, ...] = ()
    excluded_sku_ids: tuple[str, ...] = ()
    excluded_materials_by_category: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category is not None and self.category not in CATEGORIES:
            raise ValueError(f"category 必须为目录分类之一：{', '.join(CATEGORIES)}")
        if any(tag not in MATERIAL_TAGS for tag in (*self.excluded_material_tags, *self.required_material_tags)):
            raise ValueError(f"材质条件必须为目录标签之一：{', '.join(MATERIAL_TAGS)}；不能扩大或猜测未知材质")
        if any(c not in CATEGORIES or any(t not in MATERIAL_TAGS for t in tags) for c,tags in self.excluded_materials_by_category.items()):
            raise ValueError("分类材质条件不属于目录枚举")
        for name, pattern in (("product_id", r"P\d{4,}"), ("sku_id", r"P\d{4,}-S\d+")):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not re.fullmatch(pattern, value.strip().upper()):
                    raise ValueError(f"ProductSearchSpec.{name} 格式无效")
                object.__setattr__(self, name, value.strip().upper())
        if self.product_id and self.sku_id and self.sku_id.split("-S", 1)[0] != self.product_id:
            raise ValueError("product_id 与 sku_id 不属于同一商品")
        if not isinstance(self.normalized_query, str):
            raise ValueError("ProductSearchSpec.normalized_query 必须为文字")
        if not self.normalized_query.strip() and not (self.product_id or self.sku_id):
            raise ValueError("必须提供 normalized_query、product_id 或 sku_id")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 50:
            raise ValueError("ProductSearchSpec.top_k 必须为1到50的整数")
        if self.price_max_major is not None and (type(self.price_max_major) not in (int,float) or not math.isfinite(self.price_max_major) or self.price_max_major < 0):
            raise ValueError("ProductSearchSpec.price_max_major 必须是有限的非负金额")
