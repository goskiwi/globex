# -*- coding: utf-8 -*-
"""product_search_tool

商品检索工具：结构化检索入参 → CatalogSearchUseCase → 商品卡 JSON。
MainAgent 单干与 SearchAgent 派发两条路径共用同一工具实例。
工厂模式注入 UseCase 与 EventBus，模型看到的只是工具入参与返回值结构。

函数签名直接用于 LangChain 工具 schema。
"""
import json
from langchain.tools import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.domain.catalog.taxonomy import Category

from app.application.runtime.results import ToolResult, ToolResultState

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.application.runtime.request_scope import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus
from app.application.tools.shopping_operations import RequirementChange, search_requirements


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, json_schema_extra={
        "anyOf": [{"required":[field], "properties":{field:{"type":"string", "minLength":1}}}
                  for field in ("normalized_query", "product_id", "sku_id")]})
    normalized_query: str = Field(default="", description="检索词；精确 ID 查询时可以省略")
    category: Category | None = Field(default=None, description="仅用户明确限定的目录一级分类；商品类型已生效时无需猜测大类，不填则跨分类检索")
    top_k: int = Field(default=5, ge=1, le=50, strict=True, description="候选数量，1 到 50")
    product_id: str | None = Field(default=None, pattern=r"^P\d{4,}$", description="精确商品 ID")
    sku_id: str | None = Field(default=None, pattern=r"^P\d{4,}-S\d+$", description="精确 SKU；与商品 ID 同传时须属于同一商品")
    conditions: RequirementChange | None = Field(default=None, description='本轮买家明确的需求变化；省略继承。字段省略保留，null清除。子任务省略此项')

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
        conditions: RequirementChange | None = None,
        runtime: ToolRuntime = None,
    ) -> ToolResult:
        """检索符合本次需求的候选，返回SKU、规格、功能原文、价库存、限制和证据引用。
        conditions提交当前买家明确的需求变化，后端合并先前条件与长期偏好；省略则继承。
        子任务不提交conditions，使用绑定的局部条件。精确ID也受这些条件筛选；只查看资料用get_product_details。
        未命中不代表不存在；不猜分类、不放宽条件。候选匹配不等于符合全部用途，未知属性保持未知。
        SKU不能跨规格组合。price_major/currency为展示币种价，source_*及SKU原价为目录原币种；比较使用同币种。
        有目的地时附每SKU单件到手报价；多件报价及确认由对应业务操作计算。
        catalogue_coverage统计硬条件下的目录型号/平台覆盖，不代表未标属性已核实。
        """
        query = SearchQuery(normalized_query=normalized_query, category=category, top_k=top_k,
                            product_id=product_id, sku_id=sku_id, conditions=conditions)
        conditions = query.conditions
        normalized_query, category, top_k = query.normalized_query, query.category, query.top_k
        product_id, sku_id = query.product_id, query.sku_id
        snapshot_ctx = ShoppingContext.current()
        try:
            effective, state_updates = search_requirements(runtime, snapshot_ctx, conditions)
        except ValueError as err:
            return ToolResult(str(err), state=ToolResultState.ERROR, error_code='business_rejected')
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
            "product_type": policy.get('product_type'),
        }
        bus.publish(session_id, "tool.invoke", {"tool": "product_search_tool", "args": args})
        if ship_to and ship_to not in TariffSchedule(ExchangeRateTable()).supported_destinations():
            error = f"暂不支持的目的国：{ship_to}"
            return ToolResult(data=f"[error] {error}", state=ToolResultState.ERROR,
                              error_code="business_rejected", error_reason=error, state_updates=state_updates)
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
                product_type=policy.get('product_type'),
                excluded_materials_by_category=(effective or {}).get("parameters", {}).get("excluded_materials_by_category", {}),
                excluded_product_ids=tuple(effective.get("excluded_products", [])) if effective else (),
                excluded_sku_ids=tuple(effective.get("excluded_skus", [])) if effective else (),
            )
        except ValueError as err:
            return ToolResult(data=f"[error] {err}", state=ToolResultState.ERROR,
                              error_code="invalid_input", error_reason=str(err), state_updates=state_updates)
        result = await usecase.execute(spec)
        result["query_conditions"] = args
        if effective is not None:
            result["requirement_application"] = effective
            result["unverified_requirements"] = effective["unverified_requirements"]
        from datetime import datetime, timezone
        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        if evidence_store is not None and snapshot_ctx is not None:
            result["result_ref"] = await evidence_store.save(snapshot_ctx.buyer_id, session_id, "products", result)
        view = await bounded_tool_view(result, evidence_store, snapshot_ctx, kind='products')
        return ToolResult(data=result, model_data=view, state=ToolResultState.SUCCESS,
                          state_updates=state_updates, event_data={**result, 'filters': {k: v for k, v in policy.items() if k in {'product_type', 'price_max_major', 'target_currency', 'ship_to'}}})

    product_search_tool.input_model = SearchQuery
    return product_search_tool
