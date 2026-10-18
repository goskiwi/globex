"""逐 SKU 的硬条件资格；检索、详情与交付共用，不涉及召回和展示。"""
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.product import Product
from app.domain.catalog.sku import Sku
from app.domain.catalog.product_search_spec import ProductSearchSpec


def sku_constraint_issues(product: Product, sku: Sku, spec: ProductSearchSpec, rates: ExchangeRateTable) -> list[str]:
    """搜索、推荐及详情说明共用逐 SKU 资格；不同规格不能互借价格或库存。"""
    issues = []
    if product.product_id in spec.excluded_product_ids:
        issues.append("product_excluded")
    if sku.sku_id in spec.excluded_sku_ids:
        issues.append("sku_excluded")
    if sku.stock <= 0:
        issues.append("out_of_stock")
    if spec.category and product.category != spec.category:
        issues.append("category_mismatch")
    if spec.product_type and product.product_type != spec.product_type:
        issues.append('product_type_unknown' if product.product_type is None else 'product_type_mismatch')
    if set(product.material_tags) & (set(spec.excluded_material_tags) | set(spec.excluded_materials_by_category.get(product.category, []))):
        issues.append("material_excluded")
    if set(spec.required_material_tags) - set(product.material_tags):
        issues.append("material_required_missing")
    if spec.ship_to and spec.ship_to not in product.ships_to:
        issues.append("ship_to_unavailable")
    price = rates.convert(sku.price, spec.target_currency)
    if spec.price_max_major is not None and price.to_major_units() > spec.price_max_major:
        issues.append("over_price_cap")
    return issues
