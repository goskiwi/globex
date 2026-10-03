"""测试明确给定状态；语义更新由原生工具调用测试覆盖。"""
from app.application.agents.shopping_state import ShoppingWork, Filters, Choice


def work_fixture(selected=(), comparisons=(), budget=None):
    choices = [Choice(product_id=p.split('-S')[0], sku_id=p if '-S' in p else p+'-S1', quantity=1) for p in selected]
    return ShoppingWork(filters=Filters(price_max_major=budget, target_currency='CNY'),
        selections={c.sku_id: c for c in choices}, comparisons=list(comparisons)).model_dump()


def selected_lines(*sku_ids, quantity=1):
    return tuple({'product_id': s.split('-S')[0], 'sku_id': s, 'quantity': quantity} for s in sku_ids)


async def run_search(tool, *args, filters=None, **query):
    """测试夹具：先显式装配条件，再以新查询接口执行；不是产品兼容入口。"""
    from dataclasses import replace
    from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
    from app.application.agents.shopping_state import compile_search
    current = ShoppingContext.current() or ShoppingContextSnapshot('anonymous','test-buyer','zh-CN','CNY')
    effective = current.effective_search
    if effective is None or filters is not None:
        effective = compile_search(ShoppingWork(filters=filters or Filters(target_currency=current.currency)),
                                   current.preference_facts, current.currency)
    token = ShoppingContext.set(replace(current, effective_search=effective))
    try:
        return await tool(*args, **query)
    finally:
        ShoppingContext.reset(token)
