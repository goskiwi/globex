"""Main显式交付只读商品详情；不调用推荐筛选，不修改购物状态。"""
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application.tools.order_tools import _ok, _fail
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.runtime.request_scope import ShoppingContext


class ProductTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    product_id: str = Field(pattern=r"^P\d{4,}$", description="已从目录、历史或筛选拒绝摘要中识别的具体商品ID")
    sku_id: str | None = Field(default=None, pattern=r"^P\d{4,}-S\d+$", description="可选的具体SKU；卡片仍展示全部规格，不隐藏缺货规格")

    @model_validator(mode="after")
    def matching_product(self):
        if self.sku_id and self.sku_id.split('-S')[0] != self.product_id:
            raise ValueError("product_id与sku_id必须属于同一商品")
        return self


class ProductViewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    targets: list[ProductTarget] = Field(min_length=1, max_length=12, description="要查看的商品与可选具体SKU；不同平台记录分别填写，明确规格须保持，不代表购买选购")
    guidance: str = Field(min_length=1, max_length=600, description="对买家说的简短查看说明，用商品名称，不用内部编号。资料、规格、价格和库存由卡片展示，不逐项复述，不劝提高预算才能看")


def build_product_view_tool(catalog, evidence_store, bus):
    async def show_product_details(targets: list[ProductTarget], guidance: str):
        """把指定商品交付为只读详情卡，缺货、超预算或条件不满足也能展示。

        用户要求看看、查看详情时直接提交指定对象，本工具实时读取目录并展示完整资料。
        不确定对象时先从检索/历史定位目录ID。商品事实由代码实时读取，不由你重填。
        同款由页面按canonical_product_id合并为一张商品卡，卡内保留各平台全部规格和报价。
        targets仍提交各平台记录，不能自行丢掉报价；guidance不要按记录数说成多款或多张卡。
        不能用recommend_products代替详情展示。
        不变更预算、选购项或交易授权；已交付资料不再重复生成参数报告。
        """
        request = ProductViewInput(targets=targets, guidance=guidance)
        context = ShoppingContext.current()
        if context is None:
            return _fail("缺少买家会话")
        try:
            if len({(t.product_id,t.sku_id) for t in request.targets}) != len(request.targets):
                raise ValueError("详情查看对象不得重复")
            effective = context.effective_search or {"parameters": {}}
            parameters = dict(effective["parameters"])
            budget = parameters.pop("landed_budget_major", None)
            parameters["target_currency"] = parameters.get("target_currency") or context.currency
            targets_by_product = {}
            for target in request.targets:
                targets_by_product.setdefault(target.product_id, [])
                if target.sku_id:
                    targets_by_product[target.product_id].append(target.sku_id)
            cards = []
            for identifier, requested_skus in targets_by_product.items():
                spec = ProductSearchSpec(product_id=identifier, **parameters,
                    excluded_product_ids=tuple(effective.get("excluded_products", [])),
                    excluded_sku_ids=tuple(effective.get("excluded_skus", [])))
                result = await catalog.product_details(identifier, spec,
                    sku_id=requested_skus[0] if requested_skus else None, landed_budget_major=budget)
                if not result["hits"]:
                    raise ValueError(f"目录不存在该商品，不能展示：{identifier}")
                card = result['hits'][0]
                if set(requested_skus) - {sku['sku_id'] for sku in card['skus']}:
                    raise ValueError(f'指定规格不属于该商品，不能展示：{identifier}')
                if requested_skus:
                    card['requested_sku_ids'] = requested_skus
                cards.extend(result["hits"])
            payload = {"guidance": request.guidance, "hits": cards, "purpose": "product_view"}
            payload["result_ref"] = await evidence_store.save(context.buyer_id, context.shopping_session_id, "product_view", payload)
            return _ok(payload)
        except (ValueError, KeyError, TypeError) as error:
            return _fail(str(error))
    return show_product_details
