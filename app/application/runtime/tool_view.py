"""工具原文与模型视图分离；首次结果也必须在分配的容量内。"""
from contextvars import ContextVar

from app.infrastructure.context_products import product_page, token_estimate
from app.application.runtime.model_views import product_result_view


tool_view_budget: ContextVar[int | None] = ContextVar('tool_view_budget', default=None)


def research_result_view(data, *, minimal=False):
    """待交付研究保留候选事实；观察全集和明细仍由原始证据承载。"""
    view = product_result_view(data)
    scope = view.get('scope')
    if isinstance(scope, dict):
        view['scope'] = {key: value for key, value in scope.items()
                         if key not in {'parent_conditions', 'definition'}}
    for key in ('observed_products', 'observed_skus', 'historical_evidence_refs'):
        view.pop(key, None)
    refs = set()
    for candidate in view.get('candidates', []):
        facts = candidate.get('facts', {})
        if facts.get('result_ref'):
            refs.add(facts['result_ref'])
        for quote in facts.get('quotes', []):
            if quote.get('result_ref'):
                refs.add(quote['result_ref'])
            # 标题已由 product 唯一提供，不在每份报价中重抄。
            if isinstance(quote.get('line'), dict):
                quote['line'].pop('title', None)
        if minimal and facts.get('result_ref'):
            product = facts.get('product', {})
            omitted = [key for key in ('description', 'highlights') if key in product]
            if omitted:
                facts['product'] = {key: value for key, value in product.items() if key not in omitted}
                facts['offloaded'] = True
                facts['omitted_product_fields'] = omitted
    if refs and 'evidence_refs' in view:
        view['evidence_refs'] = sorted(refs)
    view['compact_view'] = True
    view['notice'] = ('候选事实和任务状态保留；未展开的观察全集及商品字段按引用回查。'
                      '待核验问题不能作为已确认属性；报价仍按其SKU、目的地、数量、币种和观察时间使用。')
    return view


async def bounded_tool_view(data, store, context, *, kind, token_limit=12000):
    """保留完整证据，以商品页或引用代替过大正文；预算包含完整包装。"""
    assigned = tool_view_budget.get()
    limit = min(token_limit, assigned) if assigned is not None else token_limit
    view = product_result_view(data)
    if token_estimate(view) <= limit:
        return view
    if kind == 'handoff':
        view = research_result_view(data)
        if token_estimate(view) <= limit:
            return view
        view = research_result_view(data, minimal=True)
        if token_estimate(view) <= limit:
            return view
    if store is None or context is None:
        raise ValueError('大型工具结果缺少证据存储，未向模型返回不完整原文')
    reference = data.get('result_ref')
    if not reference:
        reference = await store.save(context.buyer_id, context.shopping_session_id, kind, data)
        data['result_ref'] = reference
    pointer = {'result_ref': reference, 'offloaded': True, 'incomplete': True,
               'result_kind': kind, 'historical': False,
               'notice': '完整结果已保存。使用 conversation_fact_lookup 按 result_ref 分页回查；未展开部分不能当作已经读取。'}
    # 停止或待确认语义必须仍然可见，引用本身不代表任务完成或交易已执行。
    for key in ('status', 'stop_reason', 'transaction_state'):
        if key in data:
            pointer[key] = data[key]
    if token_estimate(pointer) > limit:
        raise ValueError('工具结果引用仍超过可用容量，未将超大正文写入模型历史')
    if isinstance(data.get('hits'), list):
        page_budget = limit - token_estimate(pointer)
        while page_budget > 128:
            page = product_page(data, token_limit=page_budget)
            candidate = {**page, **pointer}
            if token_estimate(candidate) <= limit:
                return candidate
            page_budget -= max(128, token_estimate(candidate) - limit)
    return pointer
