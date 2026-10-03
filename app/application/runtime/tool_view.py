"""工具原文与模型视图分离；首次结果也必须在分配的容量内。"""
from contextvars import ContextVar

from app.infrastructure.context_products import business_view, product_page, token_estimate


tool_view_budget: ContextVar[int | None] = ContextVar('tool_view_budget', default=None)


async def bounded_tool_view(data, store, context, *, kind, token_limit=12000):
    """保留完整证据，以商品页或引用代替过大正文；预算包含完整包装。"""
    assigned = tool_view_budget.get()
    limit = min(token_limit, assigned) if assigned is not None else token_limit
    view = business_view(data)
    if token_estimate(view) <= limit:
        return view
    if kind == 'handoff':
        for candidate in view.get('candidates', []):
            facts = candidate.get('facts', {})
            if facts.get('result_ref'):
                candidate['facts'] = {'result_ref': facts['result_ref'], 'historical': facts['historical'],
                    'observed_at': facts.get('observed_at'), 'offloaded': True,
                    'notice': '完整事实按本引用、商品或SKU回查；未展开资料不能猜测。'}
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
