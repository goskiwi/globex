# -*- coding: utf-8 -*-
"""product_search_tool

商品检索工具：结构化检索入参 → CatalogSearchUseCase → 商品卡 JSON。
MainAgent 单干与 SearchAgent 派发两条路径共用同一工具实例。
工厂模式注入 UseCase 与 EventBus，模型看到的只是工具入参与返回值结构。

函数签名直接用于 LangChain 工具 schema。
"""
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.domain.catalog.taxonomy import Category

from app.application.runtime.results import ToolResult, ToolResultState

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, json_schema_extra={
        "anyOf": [{"required":[field], "properties":{field:{"type":"string", "minLength":1}}}
                  for field in ("normalized_query", "product_id", "sku_id")]})
    normalized_query: str = Field(default="", description="检索词；精确 ID 查询时可以省略")
    category: Category | None = Field(default=None, description="目录一级分类，不填则不限制分类")
    top_k: int = Field(default=5, ge=1, le=50, strict=True, description="候选数量，1 到 50")
    product_id: str | None = Field(default=None, pattern=r"^P\d{4,}$", description="精确商品 ID")
    sku_id: str | None = Field(default=None, pattern=r"^P\d{4,}-S\d+$", description="精确 SKU；与商品 ID 同传时须属于同一商品")

    @model_validator(mode="after")
    def query_or_identity(self):
        if not (self.normalized_query or self.product_id or self.sku_id):
            raise ValueError("normalized_query、product_id、sku_id 至少提供一个")
        if self.product_id and self.sku_id and self.sku_id.split('-S')[0] != self.product_id:
            raise ValueError("product_id 与 sku_id 必须属于同一商品")
        return self
from app.application.runtime.tool_view import bounded_tool_view


def build_product_search_tool(usecase: CatalogSearchUseCase, bus: TradeEventBus, evidence_store=None):
    async def product_search_tool(
        normalized_query: str = "",
        category: Category | None = None,
        top_k: int = 5,
        product_id: str | None = None,
        sku_id: str | None = None,
    ) -> ToolResult:
        """按当前已生效条件筛选推荐候选，不是查看商品详情的入口。
        查看某款的完整资料、库存或当前价格用get_product_details；本工具精确ID查询仍筛选条件。
        只填写检索词、目录分类、返回数量或精确 ID；normalized_query/product_id/sku_id 至少提供一个。
        预算、币种、目的地和材质限制由当前购物状态或委派条件自动提供，不接受重复输入。
        被筛掉的商品不代表不能查看，使用filtered_out中的ID读取详情；不得为查看修改预算。
        缺少/错误的条件交回 Main 更新状态；搜索不能偷偷放宽。存在目的地时返回单件到手价，
        多件总价使用 quote_products。到手总预算在报价/最终推荐阶段校验，不等同于商品单价上限。
        分类必须使用 schema 枚举，不填则不作分类硬过滤。
        返回skus只包含本次符合硬条件且有库存的规格；默认规格是其中目标币种价格最低者，
        仅用于候选展示，不代表买家已经选择。查看全部规格仍调用get_product_details。
        规格原文参与召回，但相关性不证明颜色/尺寸等要求已满足；逐条核对真实SKU，不能组合不同规格的选项。
        filtered_out按商品提供skus中的具体规格价格和reasons，不把一条规格的价格当整件商品价格。
        候选资料含 description（用途与功能原文）、highlights（亮点）、skus（具体规格及价库存）、
        material_tags（目录材质标签）、weight_kg（每个销售单位净重，千克，未知时不返回）、
        dimensions_cm（商品尺寸，厘米）及 package_dimensions_cm（包装尺寸，不能当商品展开尺寸）。
        price_major/currency 是展示币种的商品价；source_price_major/source_currency 和 skus 中的价格保留目录原币种。
        原币种与展示币种可以不同，这本身不是数据冲突；相对价格判断使用相同币种的完整报价。
        根据返回的字段名、值和描述阅读商品，不假设存在未返回的处理器、内存或功能字段。
        候选是检索匹配，不保证适合用户；缺资料不等于没有功能，冲突字段不能用于证明优势。
        商品和 SKU 同时填写时必须属于同一商品；精确查询未命中不能用相似商品代替。
        """
        query = SearchQuery(normalized_query=normalized_query, category=category, top_k=top_k,
                            product_id=product_id, sku_id=sku_id)
        normalized_query, category, top_k = query.normalized_query, query.category, query.top_k
        product_id, sku_id = query.product_id, query.sku_id
        snapshot_ctx = ShoppingContext.current()
        effective = snapshot_ctx.effective_search if snapshot_ctx else None
        if effective is None:
            return ToolResult("缺少当前有效购物条件，请先由主 Agent 建立购物状态或派发任务。", state=ToolResultState.ERROR,
                              error_code="invalid_input", error_reason="缺少当前有效购物条件")
        policy = effective["parameters"]
        price_max_major, target_currency = policy["price_max_major"], policy["target_currency"]
        ship_to = policy["ship_to"]
        excluded_material_tags = policy["excluded_material_tags"]
        required_material_tags = policy["required_material_tags"]
        session_id = ShoppingContext.current_session_id()
        args = {
            "normalized_query": normalized_query,
            "category": category,
            "ship_to": ship_to,
            "top_k": top_k,
            "price_max_major": price_max_major,
            "target_currency": target_currency,
            "excluded_material_tags": excluded_material_tags or [],
            "required_material_tags": required_material_tags or [],
            "product_id": product_id,
            "sku_id": sku_id,
        }
        bus.publish(session_id, "tool.invoke", {"tool": "product_search_tool", "args": args})
        if ship_to and ship_to not in TariffSchedule(ExchangeRateTable()).supported_destinations():
            error = f"暂不支持的目的国：{ship_to}"
            return ToolResult(data=f"[error] {error}", state=ToolResultState.ERROR,
                              error_code="business_rejected", error_reason=error)
        try:
            spec = ProductSearchSpec(
                normalized_query=normalized_query,
                category=category,
                ship_to=ship_to,
                top_k=top_k,
                price_max_major=price_max_major,
                target_currency=target_currency,
                excluded_material_tags=excluded_material_tags or [],
                required_material_tags=required_material_tags or [],
                product_id=product_id,
                sku_id=sku_id,
                excluded_materials_by_category=(effective or {}).get("parameters", {}).get("excluded_materials_by_category", {}),
                excluded_product_ids=tuple(effective.get("excluded_products", [])) if effective else (),
                excluded_sku_ids=tuple(effective.get("excluded_skus", [])) if effective else (),
            )
        except ValueError as err:
            return ToolResult(data=f"[error] {err}", state=ToolResultState.ERROR,
                              error_code="invalid_input", error_reason=str(err))
        result = await usecase.execute(spec)
        result["query_conditions"] = args
        if effective is not None:
            result["requirement_application"] = effective
            result["unverified_requirements"] = effective["unverified_requirements"]
        from datetime import datetime, timezone
        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        if evidence_store is not None and snapshot_ctx is not None:
            result["result_ref"] = await evidence_store.save(snapshot_ctx.buyer_id, session_id, "products", result)
        result["requirement_notice"] = "单价、目的地和材质条件已用于检索；到手总预算须在报价/最终推荐阶段核验。未核验要求与软偏好不得宣称全部满足。"
        view = await bounded_tool_view(result, evidence_store, snapshot_ctx, kind='products')
        return ToolResult(data=result, model_data=view, state=ToolResultState.SUCCESS)

    product_search_tool.input_model = SearchQuery
    return product_search_tool
