"""报价只读；最终推荐由 Main 提交，商品事实与金额始终由业务代码装配。"""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.application.tools.order_tools import _verified_order_items, _ok, _fail
from app.application.catalog_evidence import verified_sku_items
from app.application.usecases.pricing import QuoteItem
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.application.runtime.request_scope import ShoppingContext
from app.application.agents.shopping_state import Choice

MAX_DECISION_ITEMS = 12


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    task_id: str | None = Field(default=None, description="计划步骤或task_dispatch返回的task_id；使用研究候选时必须填写，直接检索且无研究任务时省略")
    product_id: str = Field(min_length=1)
    sku_id: str = Field(min_length=1)
    quantity: StrictInt = Field(gt=0, title="购买数量")
    reason: str = Field(min_length=1, max_length=240, description="依据已读特点解释这款与本次需要的联系和关键取舍；不重复组件已有的数据")
    tradeoffs: list[str] = Field(default_factory=list, max_length=3, description="影响本次选择的实际局限或资料缺口；未知不等于没有，无必要取舍时为空")


class RecommendationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    picks: list[Pick] = Field(min_length=1, max_length=MAX_DECISION_ITEMS, title="推荐商品", description="本轮选择的商品，按希望展示的顺序排列")
    guidance: str = Field(min_length=1, max_length=1200, description="面向买家说明本次选择依据与必要取舍，不重复卡片数值或内部过程")


class ComparisonInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    entries: list[Pick] = Field(min_length=2, max_length=MAX_DECISION_ITEMS, title="比较商品")
    guidance: str = Field(min_length=1, max_length=1200, description="解释候选之间与本次需求有关的差异和取舍")
    preferred_sku_id: str | None = Field(description="比较中的推荐倾向；必须属于本次集合，不能据此替买家选购或交易")
    dimensions: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(max_length=6,
        description="本次比较关注点，不要求逐维填表；不适用时为空")


def _decision(request: ComparisonInput, cards: list[dict]) -> dict:
    preferred = request.preferred_sku_id
    if preferred is not None:
        card = next((card for card in cards if card['default_sku_id'] == preferred), None)
        if card is None:
            raise ValueError("首选必须属于本次交付的 SKU")
        if card['constraint_issues']:
            raise ValueError("首选不满足当前条件，只能作为对照展示")
        cards.sort(key=lambda card: card['default_sku_id'] != preferred)
    return {'guidance': request.guidance, 'preferred_sku_id': preferred, 'dimensions': request.dimensions,
            'max_items': MAX_DECISION_ITEMS}


class QuoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    items: list[Choice] = Field(min_length=1, description="每行提供已核验商品、SKU 和正整数数量")
    ship_to: str = Field(pattern=r"^[A-Z]{2}$", description="本次报价目的国，不修改购物状态")
    currency: str = Field(pattern=r"^[A-Z]{3}$", description="本次报价币种，如 CNY")


def build_quote_tool(pricing, evidence_store, bus):
    async def quote_products(items: list[Choice], ship_to: str, currency: str) :
        """查询指定 SKU、数量、目的地和币种的完整报价。不选购、不准备确认、不下单。

        items 每项必须含 product_id、sku_id、quantity；必须来自当前会话原始搜索或详情资料。
        多商品按一起购买报价；比较替代选项时分别报价。金额由代码计算。
        """
        context = ShoppingContext.current()
        if context is None:
            return _fail("缺少买家会话")
        try:
            request = QuoteInput(items=items, ship_to=ship_to, currency=currency)
            verified = await verified_sku_items(evidence_store.find_observed_product, context.buyer_id, context.shopping_session_id,
                                                   [item.model_dump() for item in request.items])
            quote = await pricing.quote([QuoteItem(x.product_id, x.sku_id, x.quantity) for x in verified],
                                        request.ship_to, request.currency)
        except (ValueError, KeyError, TypeError) as exc:
            return _fail(str(exc))
        ref = await evidence_store.save(context.buyer_id, context.shopping_session_id, "quote", quote)
        from datetime import datetime, timezone
        result = {"quote": quote, "result_ref": ref, "observed_at": datetime.now(timezone.utc).isoformat()}
        return _ok(result)
    quote_products.input_model = QuoteInput
    return quote_products


async def _recommend(catalog, evidence_store, request, *, bundle):
    context = ShoppingContext.current()
    if context is None:
        return _fail("缺少买家会话")
    try:
        cards, quotes, budget, ship_to, currency, unverified = await _assemble_choices(catalog, evidence_store, context, request.picks, bundle=bundle)
        for card in cards:
            if any(issue != "over_landed_budget" for issue in card["constraint_issues"]):
                raise ValueError(f"推荐不满足当前条件：{card['default_sku_id']} {card['constraint_issues']}")
        combined = catalog.pricing.assemble([q["items"][0] for q in quotes], ship_to, currency) if quotes and bundle else None
        checked = [combined] if combined else quotes
        if budget is not None and ship_to and any(q["total_amount_minor"] > round(budget * 100) for q in checked):
            return _fail("推荐到手总额超过当前预算，请调整选项或明确交回预算缺口。报价：" + str(checked))
        mode = 'bundle' if bundle else 'individual'
        result = {"guidance":request.guidance, "mode":mode, "hits":cards, "quote":combined,
                  "unverified_requirements":unverified}
        _attach_plan_outcome(result, context, request.picks)
        result["result_ref"] = await evidence_store.save(context.buyer_id, context.shopping_session_id, "recommendation", result)
        return _ok(result)
    except (ValueError, KeyError, TypeError) as exc:
        return _fail(str(exc))


def build_recommendation_tool(catalog, evidence_store, bus):
    async def recommend_products(picks: list[Pick], guidance: str):
        """交付本轮选定商品与建议，顺序即展示顺序。事实与金额由目录和计价服务提供。

        按买家本轮需要确定选择项，reason说明每项与实际需求的联系，guidance说明整体选择依据。
        各选择独立报价；一起购买的整套组合使用recommend_bundle。仅交付建议，不选购或交易。
        """
        return await _recommend(catalog, evidence_store, RecommendationInput(picks=picks,guidance=guidance),bundle=False)
    return recommend_products


def build_bundle_tool(catalog, evidence_store, bus):
    async def recommend_bundle(picks: list[Pick], guidance: str):
        """交付一起购买的整套商品组合；按全部规格和数量合计检查到手预算。

        reason说明每件在本次组合中的作用，guidance解释组合取舍。事实和组合总价由代码装配。
        互相替代的选择使用recommend_products。此调用不选购、不准备订单、不代表买家批准。
        """
        return await _recommend(catalog, evidence_store, RecommendationInput(picks=picks,guidance=guidance),bundle=True)
    return recommend_bundle


def build_comparison_tool(catalog, evidence_store, bus):
    async def compare_products(entries: list[Pick], preferred_sku_id: str | None, dimensions: list[str], guidance: str):
        """比较本次候选集合，至少两个 SKU；比较建议与商品一次交付，不再另调模型总结。

        reason 说明每款什么时候值得选，tradeoffs 展开这次真正影响选择的局限。
        dimensions 只标记本次权衡的关注点；费用由代码装配，不重复写一份参数报告。
        guidance 解释这次关键差异怎样影响选择，不逐行复述比较表或金额。
        数量按用户需求填写。同一商品不同规格可同时比较。不选购、不准备订单。
        条件不满足的选项可作为对照展示，但不能标为推荐倾向。
        """
        context = ShoppingContext.current()
        if context is None:
            return _fail("缺少买家会话")
        try:
            request = ComparisonInput(entries=entries, preferred_sku_id=preferred_sku_id, dimensions=dimensions, guidance=guidance)
            cards, _, budget, ship_to, _, unverified = await _assemble_choices(catalog, evidence_store, context, request.entries, require_qualified=False)
            result = {**_decision(request, cards), "hits": cards,
                      "unverified_requirements": unverified}
            _attach_plan_outcome(result, context, request.entries)
            result["result_ref"] = await evidence_store.save(context.buyer_id, context.shopping_session_id, "comparison", result)
            return _ok(result)
        except (ValueError, KeyError, TypeError) as exc:
            return _fail(str(exc))
    return compare_products


async def _assemble_choices(catalog, evidence_store, context, picks, *, require_qualified=True, bundle=False):
    """推荐与比较共享证据、目录和报价装配；条件缺口由调用方决定如何交付。"""
    if len({p.sku_id for p in picks}) != len(picks):
        raise ValueError("同一结果中的 SKU 不得重复，请合并数量")
    await _verified_order_items(evidence_store, context.buyer_id, context.shopping_session_id,
        [p.model_dump() for p in picks if p.task_id is None])
    policy = context.effective_search or {}
    params = dict(policy.get("parameters", {}))
    budget = params.pop("landed_budget_major", None)
    currency = params.get("target_currency") or context.currency
    params["target_currency"] = currency
    ship_to = params.get("ship_to")
    products = {p.product_id: p for p in await catalog.pricing.product_repo.find_by_ids(
        list(dict.fromkeys(p.product_id for p in picks)))}
    cards, quotes = [], []
    for pick in picks:
        from app.application.runtime.research_plan import resolve_candidate_scope
        task_policy = resolve_candidate_scope(context.task_plan, pick, context.source_run_id or context.source_message_id,
                                              require_qualified=require_qualified)
        product = products.get(pick.product_id)
        sku = product.find_sku(pick.sku_id) if product else None
        if sku is None or sku.stock < pick.quantity:
            raise ValueError(f"推荐规格不存在或库存不足：{pick.sku_id}")
        spec = ProductSearchSpec(normalized_query=pick.product_id, **params,
            excluded_product_ids=tuple(policy.get("excluded_products", [])),
            excluded_sku_ids=tuple(policy.get("excluded_skus", [])))
        issues = catalog.sku_constraint_issues(product, sku, spec)
        if task_policy is not None:
            task_params = dict(task_policy['parameters'])
            task_budget = task_params.pop('landed_budget_major', None)
            task_spec = ProductSearchSpec(product_id=pick.product_id, **task_params,
                excluded_product_ids=tuple(task_policy.get('excluded_products', [])),
                excluded_sku_ids=tuple(task_policy.get('excluded_skus', [])))
            task_issues = catalog.sku_constraint_issues(product, sku, task_spec)
            if task_budget is not None:
                if not task_spec.ship_to:
                    raise ValueError('子任务到手预算缺少目的地，不能报告已核验')
                task_quote = catalog.pricing.quote_observed([QuoteItem(pick.product_id, pick.sku_id, pick.quantity)],
                                                        products, task_spec.ship_to, task_spec.target_currency)
                if task_quote['total_amount_minor'] > round(task_budget * 100):
                    task_issues.append('over_landed_budget')
            if task_issues and require_qualified:
                raise ValueError('候选不满足所属任务当前条件：' + ', '.join(task_issues))
            issues = list(dict.fromkeys([*issues, *task_issues]))

        card = catalog.product_card(0, product, spec, primary=sku, skus=[sku]).to_dict()
        card.update(recommendation_reason=pick.reason, tradeoffs=pick.tradeoffs, quantity=pick.quantity)
        if ship_to:
            quote = catalog.pricing.quote_observed([QuoteItem(pick.product_id, pick.sku_id, pick.quantity)], products, ship_to, currency)
            card["landed_price"] = quote
            quotes.append(quote)
        card["constraint_issues"] = issues
        if budget is not None and ship_to and card["landed_price"]["total_amount_minor"] > round(budget * 100):
            card["constraint_issues"].append("over_landed_budget")
        cards.append(card)

    if bundle:
        _check_task_bundle_budgets(catalog.pricing, products, context, picks)
    unverified = list(policy.get("unverified_requirements", []))
    if budget is not None and not ship_to:
        unverified.append("到手预算尚未核验，当前展示商品价")
    return cards, quotes, budget, ship_to, currency, unverified


def _attach_plan_outcome(result, context, picks):
    from app.application.runtime.research_plan import plan_delivery, project_delivery_outcome
    outcome = plan_delivery(context.task_plan, picks, result["hits"])
    result['plan_outcome'] = project_delivery_outcome(outcome)
    if outcome['unmet_goals']:
        result['guidance'] += f"\n还有{len(outcome['unmet_goals'])}项关联任务尚未完成，本次仅交付已有结果。"


def _check_task_bundle_budgets(pricing, products, context, picks):
    from app.application.runtime.research_plan import resolve_candidate_scope
    for task_id in {p.task_id for p in picks if p.task_id is not None}:
        group = [p for p in picks if p.task_id == task_id]
        params = resolve_candidate_scope(context.task_plan, group[0], context.source_run_id or context.source_message_id)['parameters']
        budget = params.get('landed_budget_major')
        if budget is not None:
            quote = pricing.quote_observed([QuoteItem(p.product_id, p.sku_id, p.quantity) for p in group],
                                                products, params['ship_to'], params['target_currency'])
            if quote['total_amount_minor'] > round(budget * 100):
                raise ValueError('组合超过所属子任务的到手总预算')
