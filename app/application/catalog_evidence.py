"""商品规格来源核对；调用方明确提供只读观察或候选资格的查询策略。"""
from app.application.usecases.order_usecases import OrderItemInput


async def verified_sku_items(lookup, buyer_id: str, session_id: str, items: list[dict]) -> list[OrderItemInput]:
    """按调用方的来源权限核对规格；只读报价与候选交易不能互相授予资格。"""
    verified = []
    for item in items:
        product_id, sku_id = item.get("product_id"), item.get("sku_id")
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("商品行必须提供 product_id")
        if not isinstance(sku_id, str) or not sku_id.strip():
            raise ValueError("sku_id 必须为非空文字")
        evidence = await lookup(buyer_id, session_id, product_id=product_id, sku_id=sku_id or "")
        if evidence is None:
            raise ValueError(f"当前会话缺少可用商品证据：{sku_id or product_id}；请先取得所需来源的商品资料")
        cards = [hit for hit in evidence["data"].get("hits", []) if hit.get("product_id") == product_id]
        if not cards:
            raise ValueError("商品证据与请求不匹配")
        sku_ids = {sku["sku_id"] for card in cards for sku in card.get("skus", []) if sku.get("sku_id")}
        if sku_id not in sku_ids:
            raise ValueError(f"商品 {product_id} 缺少可核对的规格，请先读取资料并明确 sku_id")
        verified.append(OrderItemInput(product_id, sku_id, item["quantity"]))
    return verified


