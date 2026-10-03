"""指定商品详情读取：返回完整目录资料，不把当前购物条件当读取过滤器。"""
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application.runtime.results import ToolResult, ToolResultState
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext
from app.application.runtime.tool_view import bounded_tool_view


class ProductDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    product_id: str = Field(pattern=r"^P\d{4,}$", description="已从目录、历史或筛选拒绝摘要中识别的具体商品ID")
    sku_id: str | None = Field(default=None, pattern=r"^P\d{4,}-S\d+$", description="可选的具体SKU；仍返回该商品全部规格，不隐藏缺货规格")

    @model_validator(mode="after")
    def matching_product(self):
        if self.sku_id and self.sku_id.split('-S')[0] != self.product_id:
            raise ValueError("product_id与sku_id必须属于同一商品")
        return self


def build_product_details_tool(catalog, bus, evidence_store):
    async def get_product_details(product_id: str, sku_id: str | None = None) -> ToolResult:
        """查看指定商品完整资料，不修改预算、偏好、选购项或交易状态。

        用于看看某款、查规格/库存/当前价格；缺货、超预算、材质或配送不符合仍能查看。
        返回描述、亮点、尺寸、重量、平台来源及全部SKU的原币种价格、展示币种价格、库存和当前限制。
        constraint_issues只说明当前不适合推荐的原因，不代表商品不存在或不能查看；
        不能为了展示详情要求提高预算。不同product_id可能是同款不同平台，不得称为两个规格。
        不确定指代时先回查会话或用product_search_tool定位目录标识，包括filtered_out中的标识。
        本结果是详情证据，不是预算内候选或交易授权；推荐、选择及下单前仍按当前条件筛选核验。
        """
        request = ProductDetailsInput(product_id=product_id, sku_id=sku_id)
        context = ShoppingContext.current()
        if context is None:
            return ToolResult("缺少买家会话", state=ToolResultState.ERROR,
                              error_code="invalid_input", error_reason="缺少买家会话")
        effective = context.effective_search or {"parameters": {}, "unverified_requirements": []}
        params = dict(effective["parameters"])
        budget = params.pop("landed_budget_major", None)
        params["target_currency"] = params.get("target_currency") or context.currency
        spec = ProductSearchSpec(product_id=request.product_id, **params,
            excluded_product_ids=tuple(effective.get("excluded_products", [])),
            excluded_sku_ids=tuple(effective.get("excluded_skus", [])))
        bus.publish(context.shopping_session_id, "tool.invoke", {"tool": "get_product_details", "args": request.model_dump()})
        result = await catalog.product_details(request.product_id, spec, sku_id=request.sku_id,
            landed_budget_major=budget)
        result["requirement_application"] = effective
        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        result["result_ref"] = await evidence_store.save(context.buyer_id, context.shopping_session_id, "product_details", result)
        view = await bounded_tool_view(result, evidence_store, context, kind='product_details')
        return ToolResult(data=result, model_data=view, state=ToolResultState.SUCCESS)

    get_product_details.input_model = ProductDetailsInput
    return get_product_details
