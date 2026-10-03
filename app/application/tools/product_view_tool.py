"""Main显式交付只读商品详情；不调用推荐筛选，不修改购物状态。"""
from pydantic import BaseModel, ConfigDict, Field

from app.application.tools.order_tools import _ok, _fail
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext


class ProductViewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    product_ids: list[str] = Field(min_length=1, max_length=12, description="需要展示的已识别商品ID。不同平台记录分别传入；不提交候选评分或首选")
    guidance: str = Field(min_length=1, max_length=600, description="对买家说的简短查看说明，用商品名称，不用内部编号。资料、规格、价格和库存由卡片展示，不逐项复述，不劝提高预算才能看")


def build_product_view_tool(catalog, evidence_store, bus):
    async def show_product_details(product_ids: list[str], guidance: str):
        """把指定商品交付为只读详情卡，缺货、超预算或条件不满足也能展示。

        用户要求看看、查看详情时，先用get_product_details了解资料，再调用本工具展示；
        不确定对象时先从检索/历史定位目录ID。商品事实由代码实时读取，不由你重填。
        同款由页面按canonical_product_id合并为一张商品卡，卡内保留各平台全部规格和报价。
        product_ids仍提交各平台记录，不能自行丢掉报价；guidance不要按记录数说成多款或多张卡。
        不能用recommend_products代替详情展示。
        不变更预算、选购项或交易授权；成功交付后本轮结束，不再写一份参数报告。
        """
        request = ProductViewInput(product_ids=product_ids, guidance=guidance)
        context = ShoppingContext.current()
        if context is None:
            return _fail("缺少买家会话")
        try:
            if len(set(request.product_ids)) != len(request.product_ids):
                raise ValueError("详情商品ID不得重复")
            effective = context.effective_search or {"parameters": {}}
            parameters = dict(effective["parameters"])
            budget = parameters.pop("landed_budget_major", None)
            parameters["target_currency"] = parameters.get("target_currency") or context.currency
            cards = []
            for identifier in request.product_ids:
                spec = ProductSearchSpec(product_id=identifier, **parameters,
                    excluded_product_ids=tuple(effective.get("excluded_products", [])),
                    excluded_sku_ids=tuple(effective.get("excluded_skus", [])))
                selected = [line["sku_id"] for line in context.selected_lines
                            if line["product_id"] == identifier]
                # 单一明确选购规格决定详情默认项；多个规格同时选购时不擅自取舍。
                result = await catalog.product_details(identifier, spec,
                    sku_id=selected[0] if len(selected) == 1 else None,
                    landed_budget_major=budget)
                if not result["hits"]:
                    raise ValueError(f"目录不存在该商品，不能展示：{identifier}")
                if len(selected) == 1:
                    result["hits"][0]["selected_sku_id"] = selected[0]
                cards.extend(result["hits"])
            names = {identifier: card['title'] for card in cards
                     for identifier in (card['product_id'], card['default_sku_id'])}
            if any(identifier in request.guidance for identifier in names):
                raise ValueError("查看说明使用商品名称，不使用内部编号：" + '；'.join(names.values()))
            payload = {"guidance": request.guidance, "hits": cards, "purpose": "product_view"}
            payload["result_ref"] = await evidence_store.save(context.buyer_id, context.shopping_session_id, "product_view", payload)
            bus.publish(context.shopping_session_id, "product_view.result", payload)
            return _ok(payload)
        except (ValueError, KeyError, TypeError) as error:
            return _fail(str(error))
    return show_product_details
