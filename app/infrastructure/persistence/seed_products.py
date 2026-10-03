"""只读取当前版本化商品目录，不保留硬编码商品生成或旧数据转换入口。"""
from __future__ import annotations

import json
from pathlib import Path

from app.domain.catalog.money import Money
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.sku import Sku

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PACKAGED_CATALOG_FIXTURE = _PROJECT_ROOT / "catalog" / "catalog-v3.jsonl"
_DEVELOPMENT_CATALOG_FIXTURE = _PROJECT_ROOT / "data" / "catalog-v3.jsonl"
_CATALOG_FIXTURE = _PACKAGED_CATALOG_FIXTURE if _PACKAGED_CATALOG_FIXTURE.exists() else _DEVELOPMENT_CATALOG_FIXTURE


def _sku(sku_id: str, spec: str, major: float, currency: str, stock: int) -> Sku:
    return Sku(sku_id=sku_id, spec=spec, price=Money.from_major_units(major, currency), stock=stock)


def _product_from_record(record: dict) -> Product:
    """把版本化 JSONL 记录还原为领域对象。"""
    return Product(
        product_id=record["product_id"],
        title=record["title"],
        brand=record["brand"],
        category=record["category"],
        origin_country=record["origin_country"],
        description=record["description"],
        highlights=[ProductHighlight(item["label"], item.get("detail", "")) for item in record.get("highlights", [])],
        ships_to=list(record.get("ships_to", [])),
        skus=[
            _sku(item["sku_id"], item["spec"], item["price_major"], item["currency"], item["stock"])
            for item in record["skus"]
        ],
        source_platform=record.get("source_platform", ""),
        external_product_id=record.get("external_product_id", ""),
        canonical_product_id=record.get("canonical_product_id", ""),
        material_tags=list(record.get("material_tags", [])),
        weight_kg=record["weight_kg"],
        dimensions_cm=dict(record["dimensions_cm"]),
        package_dimensions_cm=dict(record["package_dimensions_cm"]),
        tax_category=record.get("tax_category", ""),
        rating_summary=record.get("rating_summary"),
        updated_at=record.get("updated_at", ""),
        source_language=record.get("source_language", ""),
        source_locale=record.get("source_locale", ""),
        localized_category=record.get("localized_category", ""),
        localized_material=record.get("localized_material", ""),
        data_provenance=record.get("data_provenance", ""),
    )


def build_seed_products() -> list[Product]:
    """加载版本化评测商品集；缺失时明确失败，避免悄悄退回过小的旧种子。"""
    if not _CATALOG_FIXTURE.exists():
        raise RuntimeError(f"商品评测数据集不存在：{_CATALOG_FIXTURE}")
    records = [
        json.loads(line)
        for line in _CATALOG_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [_product_from_record(record) for record in records]
