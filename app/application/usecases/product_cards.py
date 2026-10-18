"""商品展示投影：接收明确的规格集合，费用统一来自计价服务。"""
from dataclasses import dataclass, field
from typing import Optional

from app.domain.catalog.product import Product
from app.domain.catalog.sku import Sku
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.usecases.pricing import PricingService
from app.application.usecases.product_media import product_media

@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    price_major: float
    currency: str
    source_price_major: float
    source_currency: str
    highlights: list[str]
    skus: list[dict]
    score: float
    landed_price: Optional[dict]  # ship_to 命中时的到手价明细，未命中为 None
    source_platform: str
    canonical_product_id: str
    material_tags: list[str]
    weight_kg: float | None
    # 展示资料与具体默认 SKU 的价库存一起组装。
    description: str = ""
    rating_summary: dict[str, float | int] | None = None
    rating_is_live: bool = False  # 当前目录是评测快照，评分不来自实时平台查询。
    ships_to: list[str] = field(default_factory=list)
    dimensions_cm: dict[str, float] = field(default_factory=dict)
    package_dimensions_cm: dict[str, float] = field(default_factory=dict)
    updated_at: str = ""
    default_sku_id: str = ""
    image_url: str | None = None
    image_kind: str = "placeholder"
    image_alt: str = "暂无商品图片"
    source_language: str = ""
    source_locale: str = ""
    data_provenance: str = ""

    def to_dict(self) -> dict:
        card = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "origin_country": self.origin_country,
            "price_major": self.price_major,
            "currency": self.currency,
            "source_price_major": self.source_price_major,
            "source_currency": self.source_currency,
            "highlights": self.highlights,
            "skus": self.skus,
            "score": round(self.score, 4),
            "source_platform": self.source_platform,
            "canonical_product_id": self.canonical_product_id,
            "material_tags": self.material_tags,
            "description": self.description,
            "rating_summary": self.rating_summary,
            "rating_is_live": self.rating_is_live,
            "ships_to": self.ships_to,
            "dimensions_cm": self.dimensions_cm,
            "package_dimensions_cm": self.package_dimensions_cm,
            "updated_at": self.updated_at,
            "default_sku_id": self.default_sku_id,
            "image_url": self.image_url,
            "image_kind": self.image_kind,
            "image_alt": self.image_alt,
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        if self.weight_kg is not None:
            card["weight_kg"] = self.weight_kg
        if self.source_language:
            card.update(source_language=self.source_language, source_locale=self.source_locale)
        if self.data_provenance:
            card["data_provenance"] = self.data_provenance
        return card


def assemble_product_card(score: float, product: Product, spec: ProductSearchSpec, *,
                          primary: Sku, skus: list[Sku], pricing: PricingService) -> ProductCard:
    media = product_media(product.product_id, product.title)
    primary_in_target = pricing.tariff.rates.convert(primary.price, spec.target_currency)
    landed_price: Optional[dict] = None
    if spec.ship_to:
        try:
            line = pricing.price_sku(product, primary, 1, spec.ship_to, spec.target_currency)
            landed_price = pricing.assemble([line], spec.ship_to, spec.target_currency)
        except ValueError as err:
            # 目的国不在规则表内：如实标注，不编造数字
            landed_price = {"unavailable_reason": str(err)}
    return ProductCard(
        product_id=product.product_id,
        title=product.title,
        brand=product.brand,
        category=product.category,
        origin_country=product.origin_country,
        # 商品卡价格必须与预算过滤使用同一目标币种；原始报价单独保留，
        # SKU 详情继续维持平台原币种，方便后续下单时选择具体规格。
        price_major=primary_in_target.to_major_units(),
        currency=spec.target_currency,
        source_price_major=primary.price.to_major_units(),
        source_currency=primary.price.currency,
        highlights=[f"{h.label}：{h.detail}" if h.detail else h.label for h in product.highlights],
        skus=[
            {
                "sku_id": sku.sku_id,
                "spec": sku.spec,
                "price_major": sku.price.to_major_units(),
                "currency": sku.price.currency,
                "stock": sku.stock,
            }
            for sku in skus
        ],
        score=score,
        landed_price=landed_price,
        source_platform=product.source_platform,
        canonical_product_id=product.canonical_product_id,
        material_tags=product.material_tags,
        weight_kg=product.weight_kg,
        description=product.description,
        rating_summary=dict(product.rating_summary) if product.rating_summary is not None else None,
        ships_to=list(product.ships_to),
        dimensions_cm=dict(product.dimensions_cm),
        package_dimensions_cm=dict(product.package_dimensions_cm),
        updated_at=product.updated_at,
        source_language=product.source_language,
        source_locale=product.source_locale,
        data_provenance=product.data_provenance,
        default_sku_id=primary.sku_id,
        image_url=media.image_url,
        image_kind=media.image_kind,
        image_alt=media.image_alt,
    )
