"""当前检索条件和按 SKU 保存的选购项；不保存报价或交易授权。"""
from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer
from app.application.memory.preference_selector import preference_constraints
from app.domain.catalog.taxonomy import MaterialTag


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Filters(StrictModel):
    price_max_major: float | None = Field(default=None, ge=0, description="单件商品价上限，不含运税；到手总预算使用 landed_budget_major")
    landed_budget_major: float | None = Field(default=None, ge=0, description="本次选购的到手总预算，包含运税；组合按合计，备选按各选项比较")
    target_currency: str | None = None
    ship_to: str | None = Field(default=None, min_length=2, max_length=2)
    excluded_material_tags: list[MaterialTag] = Field(default_factory=list)
    required_material_tags: list[MaterialTag] = Field(default_factory=list)


class PlanStep(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,39}$")
    goal: str = Field(min_length=1, max_length=240)
    filters: Filters = Field(default_factory=Filters)
    requirements: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @model_serializer(mode='wrap')
    def serialize(self, handler):
        value = handler(self)
        value['filters'] = self.filters.model_dump(exclude_unset=True)
        return value


def validate_plan(steps):
    seen = set()
    for step in steps:
        if step.id in seen or set(step.depends_on) - seen:
            raise ValueError("计划步骤ID必须唯一；依赖必须引用排在前面的步骤，不能有循环")
        seen.add(step.id)
    return steps


class Choice(StrictModel):
    product_id: str = Field(pattern=r"^P\d{4,}$")
    sku_id: str = Field(pattern=r"^P\d{4,}-S\d+$")
    quantity: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def matching_product(self):
        if self.sku_id.split('-S')[0] != self.product_id:
            raise ValueError("商品与 SKU 不匹配")
        return self


class ShoppingWork(StrictModel):
    filters: Filters  # 必填以拒绝旧 checkpoint，不转换旧多目标状态。
    goal: str = ""
    plan: list[PlanStep] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    sort: str | None = None
    ignored_preferences: list[str] = Field(default_factory=list)
    unverified_requirements: list[str] = Field(default_factory=list)
    selections: dict[str, Choice] = Field(default_factory=dict)
    comparisons: list[str] = Field(default_factory=list)
    excluded_products: list[str] = Field(default_factory=list)
    excluded_skus: list[str] = Field(default_factory=list)
    source_message_id: str = ""
    latest_request: str = ""


class ShoppingUpdate(StrictModel):
    plan: list[PlanStep] | None = Field(default=None, max_length=12, description="复杂多目标任务的研究计划；步骤只填目标、局部条件、依赖，状态由执行证据决定。修改时提交完整计划；[]清空。单步查询不建计划。")
    reset: bool = Field(default=False, description="明确开始新的选购任务才清空当前条件及选择")
    goal: str | None = Field(default=None, description="本轮选购目标，不填则保留；不是检索词或商品事实")
    filters: Filters | None = Field(default=None, description="只更新提供的字段；null 清空全部过滤，空列表清空该材质条件")
    preferences: list[str] | None = Field(default=None, description="用户表达的软偏好，不作为硬过滤；空列表清空")
    sort: str | None = None
    ignored_preferences: list[str] | None = Field(default=None, description="本次不采用的长期偏好原文；[] 恢复全部；不修改长期记忆")
    unverified_requirements: list[str] | None = None
    selections: list[Choice] = Field(default_factory=list, description="按 SKU 新增或替换数量；不同 SKU 独立保存。必须来自会话检索证据")
    remove_skus: list[str] = Field(default_factory=list)
    comparisons: list[str] | None = Field(default=None, description="用户要求比较的对象或品类名称；不填保留，空列表清空")
    excluded_products: list[str] | None = None
    excluded_skus: list[str] | None = None


def apply_update(previous: ShoppingWork, update: ShoppingUpdate, source_message_id: str, preferences=()) -> ShoppingWork:
    # 来源由当前已认证买家消息绑定；模型不能填写或替换来源，不靠复制原文证明授权。
    if not source_message_id or source_message_id != previous.source_message_id:
        raise ValueError("购物状态修改没有绑定当前买家输入")
    work = ShoppingWork(filters=Filters(), latest_request=previous.latest_request,
                        source_message_id=previous.source_message_id) if update.reset else previous.model_copy(deep=True)
    if 'filters' in update.model_fields_set:
        work.filters = Filters() if update.filters is None else Filters.model_validate({
            **work.filters.model_dump(), **update.filters.model_dump(exclude_unset=True)})
    if 'plan' in update.model_fields_set:
        work.plan = validate_plan(update.plan or [])
    for key in ('goal', 'preferences', 'sort', 'ignored_preferences', 'unverified_requirements',
                'comparisons', 'excluded_products', 'excluded_skus'):
        if key in update.model_fields_set:
            value = getattr(update, key)
            setattr(work, key, value if value is not None else (None if key == 'sort' else '' if key == 'goal' else []))
    if set(work.ignored_preferences) - {p.statement for p in preferences}:
        raise ValueError("本次偏好例外必须引用已有长期偏好原文")
    for sku in update.remove_skus:
        work.selections.pop(sku, None)
    for choice in update.selections:
        if choice.product_id in work.excluded_products or choice.sku_id in work.excluded_skus:
            raise ValueError("所选规格已排除，请先明确撤销排除")
        work.selections[choice.sku_id] = choice
    work.selections = {sku: c for sku, c in work.selections.items()
                       if sku not in work.excluded_skus and c.product_id not in work.excluded_products}
    return work


def protected_products(work: dict | None) -> set[str]:
    if not work:
        return set()
    state = ShoppingWork.model_validate(work)
    return {*state.comparisons, *state.selections, *(c.product_id for c in state.selections.values())}


def compile_search(work: ShoppingWork, preferences=(), currency="CNY") -> dict:
    effective = [p for p in preferences if p.statement not in work.ignored_preferences]
    global_tags, scoped, pending = preference_constraints(effective)
    params = work.filters.model_dump()
    params['target_currency'] = params['target_currency'] or currency
    params['excluded_material_tags'] = list(dict.fromkeys([
        *global_tags, *params['excluded_material_tags']]))
    params["excluded_materials_by_category"] = scoped
    unverified = [*work.unverified_requirements, *pending]
    if params['price_max_major'] is not None and not work.filters.target_currency:
        unverified.append('预算币种未明确，尚未执行价格上限')
        params['price_max_major'] = None
    if params['landed_budget_major'] is not None and not work.filters.target_currency:
        unverified.append('到手预算币种未明确，尚未执行总价上限')
        params['landed_budget_major'] = None
    return {'parameters': params, 'unverified_requirements': unverified,
            'preferences': work.preferences, 'sort': work.sort,
            'long_term_preferences': [{'kind': p.kind, 'statement': p.statement} for p in effective],
            'excluded_products': work.excluded_products, 'excluded_skus': work.excluded_skus}
