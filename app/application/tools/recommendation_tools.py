"""报价只读；最终推荐由 Main 提交，商品事实与金额始终由业务代码装配。"""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.application.tools.order_tools import _verified_order_items, _ok, _fail
from app.application.usecases.pricing import QuoteItem
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext
from app.application.agents.shopping_state import Choice

MAX_DECISION_ITEMS = 12


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    product_id: str = Field(min_length=1)
    sku_id: str = Field(min_length=1)
    quantity: StrictInt = Field(gt=0, title="购买数量")
    reason: str = Field(min_length=1, max_length=240, description="这款为什么适合本次需要：用一到两句自然的话，把已知商品特点与使用场景联系起来，必要时说一个关键取舍。不是参数清单，不复述具体金额、库存或执行过程。备选突出与首选不同的选择理由；比较级须核对本次集合，未说明的功能不能说没有")
    tradeoffs: list[str] = Field(default_factory=list, max_length=3, description="只填写商品资料明确支持、且会改变本次选择的局限或关键资料缺口，无明确局限填[]，不用给每款凑缺点。每条只说一个取舍；不能仅凭类型或材质推断软硬、舒适、质量、压缩能力或具体使用效果，未知不等于没有。资料缺口说明尚不能确认的能力，不能改写成肯定缺点。不要写与需要无关的缺失属性、价格明细或执行过程")


class DecisionInput(BaseModel):
    """两种交付共用选择语义；不把用户需求映射成固定品类属性。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    guidance: str = Field(min_length=1, max_length=1200, description="对买家说的整体选购建议，随本次商品选择一起交付。普通选购用三到五句自然的话说明为什么首选、备选分别适合什么优先项；组合说明搭配思路。只解释本次交付的商品和规格，用title或清楚的名称简称，不用内部编号。不写具体报价、库存数量、未交付规格或候选，价格和规格由卡片说明；不逐款抄卡片理由，不复述内部过程或用‘请看卡片’占位。判断只依据明确商品资料，不从类型或材质猜测本款的软硬、舒适、质量、压缩能力或具体使用效果")
    preferred_sku_id: str | None = Field(description="明确首选的 SKU，必须属于本次集合；证据不足无法选择时填 null，理由说明缺口；组合购买填 null。这不是买家选购或交易授权")
    dimensions: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(max_length=6,
        description="本次根据用户用途与偏好实际比较的关注点，如收纳、背负；开放文本，不是属性名或评分公式。不适用时填 []，不凑维度")


class RecommendationInput(DecisionInput):
    picks: list[Pick] = Field(min_length=1, max_length=MAX_DECISION_ITEMS, title="推荐商品")
    mode: Literal["alternatives", "bundle"] = Field(description="alternatives 各选项独立报价；bundle 为一起购买的组合")


class ComparisonInput(DecisionInput):
    entries: list[Pick] = Field(min_length=2, max_length=MAX_DECISION_ITEMS, title="比较商品")


def _decision(request: DecisionInput, cards: list[dict]) -> dict:
    # 标识字段用于工具操作，买家文案使用目录名称。校验实际引用，不猜编号格式或替换原文。
    public_text = [request.guidance, *request.dimensions,
                   *(text for card in cards for text in [card['recommendation_reason'], *card['tradeoffs']])]
    references = {identifier: card['title'] for card in cards
                  for identifier in (card['product_id'], card['default_sku_id'])}
    misused = {identifier: title for identifier, title in references.items()
               if any(identifier in text for text in public_text)}
    if misused:
        names = '；'.join(f'{identifier} 对应 {title}' for identifier, title in misused.items())
        raise ValueError(f'选购建议、单款理由和关注点请使用商品名称，不展示内部编号。请修正文案后重新提交：{names}')
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

        items 每项必须含 product_id、sku_id、quantity；必须来自当前会话检索结果。
        多商品按一起购买报价；比较替代选项时分别报价。金额由代码计算。
        """
        context = ShoppingContext.current()
        if context is None:
            return _fail("缺少买家会话")
        try:
            request = QuoteInput(items=items, ship_to=ship_to, currency=currency)
            verified = await _verified_order_items(evidence_store, context.buyer_id, context.shopping_session_id,
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


def build_recommendation_tool(catalog, evidence_store, bus):
    async def recommend_products(picks: list[Pick], mode: str, preferred_sku_id: str | None, dimensions: list[str], guidance: str):
        """把本轮选择交付为商品卡。商品事实和费用由代码填充，你提交选择判断。

        检索候选不是最终推荐。推荐使用当前购物条件，不修改买家的选购项或交易授权。
        alternatives 表示互相替代的选项，bundle 表示一起购买；目的地未知时仅展示商品价。
        reason 是对买家说的短建议：首选抓住本次最重要的需要，备选说明更看重什么时值得选。
        参数只在能解释选择时引用，不逐项复述；关键的未知能力不能写成确定适配。
        alternatives 明确首选；bundle 推荐整套组合，preferred_sku_id 填 null。
        guidance 提交整体选购建议，解释首选与备选之间怎么选；reason 只解释当前单款。
        dimensions 是本次关注点，不要求逐维填表。建议与卡片一次交付，成功后结束，不再另调模型总结。
        """
        context = ShoppingContext.current()
        if context is None:
            return _fail("缺少买家会话")
        try:
            request = RecommendationInput(picks=picks, mode=mode, preferred_sku_id=preferred_sku_id, dimensions=dimensions, guidance=guidance)
            if request.mode == 'bundle' and request.preferred_sku_id is not None:
                raise ValueError("组合购买推荐整套组合，不指定单品首选")
            cards, quotes, budget, ship_to, currency, unverified = await _assemble_choices(catalog, evidence_store, context, request.picks)
            for card in cards:
                if any(issue != "over_landed_budget" for issue in card["constraint_issues"]):
                    raise ValueError(f"推荐不满足当前条件：{card['default_sku_id']} {card['constraint_issues']}")
            bundle = catalog.pricing.assemble([q["items"][0] for q in quotes], ship_to, currency) if quotes and mode == "bundle" else None
            checked = [bundle] if bundle else quotes
            if budget is not None and ship_to:
                if any(q["total_amount_minor"] > round(budget * 100) for q in checked):
                    return _fail("推荐到手总额超过当前预算，请调整选项或明确交回预算缺口。报价：" + str(checked))
            result = {**_decision(request, cards), "mode": mode, "hits": cards, "quote": bundle,
                      "unverified_requirements": unverified}
            ref = await evidence_store.save(context.buyer_id, context.shopping_session_id, "recommendation", result)
            result["result_ref"] = ref
            bus.publish(context.shopping_session_id, "recommendation.result", result)
            return _ok(result)
        except (ValueError, KeyError, TypeError) as exc:
            return _fail(str(exc))
    return recommend_products


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
            cards, _, budget, ship_to, _, unverified = await _assemble_choices(catalog, evidence_store, context, request.entries)
            result = {**_decision(request, cards), "hits": cards,
                      "unverified_requirements": unverified}
            result["result_ref"] = await evidence_store.save(context.buyer_id, context.shopping_session_id, "comparison", result)
            bus.publish(context.shopping_session_id, "comparison.result", result)
            return _ok(result)
        except (ValueError, KeyError, TypeError) as exc:
            return _fail(str(exc))
    return compare_products


async def _assemble_choices(catalog, evidence_store, context, picks):
    """推荐与比较共享证据、目录和报价装配；条件缺口由调用方决定如何交付。"""
    if len({p.sku_id for p in picks}) != len(picks):
        raise ValueError("同一结果中的 SKU 不得重复，请合并数量")
    await _verified_order_items(evidence_store, context.buyer_id, context.shopping_session_id,
        [p.model_dump() for p in picks])
    policy = context.effective_search or {}
    params = dict(policy.get("parameters", {}))
    budget = params.pop("landed_budget_major", None)
    currency = params.get("target_currency") or context.currency
    params["target_currency"] = currency
    ship_to = params.get("ship_to")
    cards, quotes = [], []
    for pick in picks:
        product = await catalog.pricing.product_repo.find_by_id(pick.product_id)
        sku = product.find_sku(pick.sku_id) if product else None
        if sku is None or sku.stock < pick.quantity:
            raise ValueError(f"推荐规格不存在或库存不足：{pick.sku_id}")
        spec = ProductSearchSpec(normalized_query=pick.product_id, **params,
            excluded_product_ids=tuple(policy.get("excluded_products", [])),
            excluded_sku_ids=tuple(policy.get("excluded_skus", [])))
        issues = catalog.sku_constraint_issues(product, sku, spec)

        card = catalog.product_card(0, product, spec, primary=sku, skus=[sku]).to_dict()
        card.update(recommendation_reason=pick.reason, tradeoffs=pick.tradeoffs, quantity=pick.quantity)
        if ship_to:
            quote = await catalog.pricing.quote([QuoteItem(pick.product_id, pick.sku_id, pick.quantity)], ship_to, currency)
            card["landed_price"] = quote
            quotes.append(quote)
        card["constraint_issues"] = issues
        if budget is not None and ship_to and card["landed_price"]["total_amount_minor"] > round(budget * 100):
            card["constraint_issues"].append("over_landed_budget")
        cards.append(card)

    unverified = list(policy.get("unverified_requirements", []))
    if budget is not None and not ship_to:
        unverified.append("到手预算尚未核验，当前展示商品价")
    return cards, quotes, budget, ship_to, currency, unverified
