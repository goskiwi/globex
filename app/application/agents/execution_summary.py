"""执行过程只投影既有参数与原生回执，不解析模型正文、不生成新的选购判断。"""
import math

TOOL_LABELS = {
    'product_search_tool': '检索商品', 'get_product_details': '读取商品资料',
    'show_product_details': '准备商品详情', 'update_shopping_state': '更新选购条件',
    'quote_products': '核对到手报价', 'recommend_products': '准备最终推荐',
    'compare_products': '准备比较结果', 'show_shopping_form': '补充选购需求',
    'SubagentResult': '提交研究结果', 'category_insight_tool': '查询选购知识',
    'web_search_tool': '核实外部资料', 'task_dispatch': '委派专家任务',
    'remember_preference_tool': '保存购物偏好', 'forget_preference_tool': '删除购物偏好',
    'update_preference_tool': '修改购物偏好', 'create_order_tool': '准备下单确认',
    'query_order_tool': '查询订单', 'cancel_order_tool': '准备取消确认',
    'load_agent_skill_tool': '读取选购方案', 'lookup_strategy_memory_tool': '查阅选购建议',
}
DELIVERY_TOOLS = {'recommend_products': 'recommendation', 'compare_products': 'comparison',
                  'show_product_details': 'product_view'}
COUNTRIES = {'CN': '中国', 'US': '美国', 'JP': '日本', 'SG': '新加坡', 'EU': '欧盟'}
AGENTS = {'search_agent': '商品研究', 'trade_agent': '交易准备'}
PLATFORMS = {'amazon': 'Amazon', 'ebay': 'eBay'}


def text(value):
    return value.strip() if isinstance(value, str) else ''


def money(amount, currency):
    if type(amount) not in (float, int) or not math.isfinite(amount) or not text(currency):
        return ''
    return f'¥{amount:.2f}' if currency == 'CNY' else f'{amount:.2f} {currency}'


def destination(value):
    return COUNTRIES.get(value, text(value)) if isinstance(value, str) else ''


def cards(data):
    return [p for p in data.get('hits', []) if isinstance(p, dict)] if isinstance(data.get('hits'), list) else []


def product_count(items):
    return len({p.get('canonical_product_id') or p.get('product_id') for p in items if p.get('canonical_product_id') or p.get('product_id')})


def label(name, arguments, data=None):
    if name == 'task_dispatch':
        agent = (data or {}).get('agent') if isinstance(data, dict) else None
        return AGENTS.get(text(agent) or text(arguments.get('subagent_type')), '委派专家任务')
    if name == 'get_product_details' and isinstance(data, dict):
        found = cards(data)
        title = text(found[0].get('title')) if len(found) == 1 else ''
        if title:
            return f'读取 {title} 的资料'
    return TOOL_LABELS.get(name, '执行工具')


def conditions(arguments):
    update = arguments.get('update')
    if not isinstance(update, dict):
        return ''
    filters = update.get('filters')
    parts = []
    if isinstance(filters, dict):
        currency = text(filters.get('target_currency'))
        for key, caption in [('landed_budget_major', '到手预算'), ('price_max_major', '商品价上限')]:
            amount = filters.get(key)
            if type(amount) in (int, float) and math.isfinite(amount):
                parts.append(f'{caption} {amount:g}' + (f' {currency}' if currency else ''))
        if text(filters.get('ship_to')):
            parts.append('配送至' + destination(filters['ship_to']))
        for key, caption in [('required_material_tags', '要求材质'), ('excluded_material_tags', '排除材质')]:
            values = filters.get(key)
            if isinstance(values, list) and values:
                parts.append(caption + '：' + '、'.join(v for v in values if isinstance(v, str)))
    selections = update.get('selections')
    if isinstance(selections, list) and selections:
        parts.append(f'更新了{len(selections)}个规格的选购数量')
    preferences=update.get('preferences')
    if isinstance(preferences,list) and preferences:
        parts.append('记录偏好：'+'、'.join(v for v in preferences if isinstance(v,str)))
    return '；'.join(parts)


def request_summary(name, arguments):
    if name == 'update_shopping_state':
        return conditions(arguments)
    if name == 'product_search_tool' and text(arguments.get('normalized_query')):
        return '检索词：' + arguments['normalized_query']
    if name == 'quote_products':
        parts = ['配送至' + destination(arguments['ship_to'])] if text(arguments.get('ship_to')) else []
        if text(arguments.get('currency')):
            parts.append('报价币种：' + arguments['currency'])
        return '；'.join(parts)
    return ''


def quote_summary(quote):
    if not isinstance(quote, dict):
        return ''
    amount = quote.get('total_amount_minor')
    value = money(amount / 100, quote.get('currency')) if type(amount) in (float, int) else ''
    lines = quote.get('items')
    parts = []
    if isinstance(lines, list):
        if len(lines) == 1 and isinstance(lines[0], dict):
            item = lines[0]
            title = text(item.get('title'))
            qty = item.get('quantity')
            if title:
                parts.append(title + (f' ×{qty}' if type(qty) is int else ''))
        elif len(lines) > 1:
            parts.append(f'{len(lines)}项商品一起购买')
    if text(quote.get('ship_to')):
        parts.append('寄' + destination(quote['ship_to']))
    if value:
        parts.append('组合到手价 ' + value if isinstance(lines, list) and len(lines) > 1 else '到手价 ' + value)
    return '；'.join(parts)


def delivery_summary(name, data, *, delivered=False):
    items = cards(data)
    if not items:
        return ''
    if name == 'show_product_details':
        platforms = {p['source_platform'] for p in items if text(p.get('source_platform'))}
        return f'{"已展示" if delivered else "已准备"}{product_count(items)}款商品的完整资料' + (f'，覆盖{len(platforms)}个平台' if platforms else '')
    kind = '推荐' if name == 'recommend_products' else '比较'
    parts = [f'{"已交付" if delivered else "已装配"}{len(items)}款{kind}商品']
    preferred = data.get('preferred_sku_id')
    primary = next((p for p in items if preferred and p.get('default_sku_id') == preferred), None)
    if primary and text(primary.get('title')):
        parts.append('首选：' + primary['title'])
    if name == 'recommend_products' and isinstance(data.get('quote'), dict):
        quoted = quote_summary(data['quote'])
        if quoted:
            parts.append(quoted)
    else:
        amounts = []
        for p in items[:3]:
            q = p.get('landed_price')
            if isinstance(q, dict) and type(q.get('total_amount_minor')) in (int, float):
                value = money(q['total_amount_minor'] / 100, q.get('currency'))
                if value and text(p.get('title')):
                    amounts.append(p['title'] + ' ' + value)
        if amounts:
            parts.append('到手报价：' + '、'.join(amounts) + (f' 等{len(items)}款' if len(items) > 3 else ''))
    return '；'.join(parts)


def failure_summary(failure):
    """界面映射错误类型，不读取异常文字或原始参数。"""
    failure = failure if isinstance(failure, dict) else {}
    code = failure.get('code')
    if code == 'invalid_input':
        issues = failure.get('issues', [])
        captions = {'missing':'缺少必填字段', 'extra_forbidden':'不接受此字段',
                    'int_type':'必须填写整数', 'int_parsing':'必须填写整数',
                    'greater_than':'未满足最小值限制', 'literal_error':'不属于允许的选项'}
        parts = []
        for issue in issues:
            path = text(issue.get('label')) or '.'.join(map(str,issue['path'])) or '参数组合'
            parts.append(path + '：' + captions.get(issue['type'],'不符合字段约束'))
        reason = text(failure.get('reason'))
        return '参数校验未通过' + ('：' + ('；'.join(parts) or reason) if parts or reason else '') + ('；本次未执行' if failure.get('executed') is False else '')
    reason = text(failure.get('reason'))
    if code == 'business_rejected':
        return '业务条件未满足' + ('：' + reason if reason else '，本次操作未完成')
    if code == 'unavailable':
        return reason or '外部服务不可用，执行结果尚未确认'
    if code == 'internal':
        return reason or '工具执行异常，执行结果尚未确认'
    if code in {'interrupted','stopped'}:
        return '执行已停止，未取得完整结果'
    return '未取得明确的失败回执，执行结果尚未确认'


def result_summary(name, arguments, data, success, *, failure=None):
    if name in {'task_dispatch','SubagentResult'} and isinstance(data,dict):
        captions = {'completed': '任务已完成', 'partial': '任务部分完成',
                    'needs_input': '需要补充信息', 'failed': '任务未完成'}
        parts = [captions.get(text(data.get('status')), '已返回任务结果')]
        if isinstance(data.get('candidates'), list):
            parts.append(f'交回{len(data["candidates"])}个研究候选')
        if data.get('transaction_state') == 'awaiting_confirmation':
            parts.append('确认单已准备，尚未执行交易')
        return '；'.join(parts)
    if not success:
        return failure_summary(failure)
    if name == 'update_shopping_state':
        return conditions(arguments) or '选购条件与选购项已更新'
    if not isinstance(data, dict):
        return '工具已返回结果'
    if name in DELIVERY_TOOLS:
        return delivery_summary(name, data)
    if name == 'product_search_tool':
        if isinstance(data.get('hits'), list):
            query=request_summary(name,arguments)
            return (query+'；' if query else '')+f'本次返回{len(data["hits"])}个候选，供进一步核对与比较'
    if name == 'get_product_details':
        items = cards(data)
        if not items:
            return '未读取到指定商品资料'
        item = items[0]
        parts = []
        if text(item.get('source_platform')):
            parts.append('平台：' + PLATFORMS.get(item['source_platform'],item['source_platform']))
        values = item.get('highlights')
        if isinstance(values, list):
            parts.extend(text(value) for value in values[:2] if text(value))
        skus = item.get('skus')
        if isinstance(skus, list):
            parts.append(f'{len(skus)}个规格，含价格与库存')
        return '；'.join(parts) or '已读取商品资料'
    if name == 'quote_products':
        return quote_summary(data.get('quote')) or '已取得报价结果'
    if name == 'load_agent_skill_tool':
        return '已读取：' + text(data.get('title')) if text(data.get('title')) else '已读取选购方案'
    confirmation = data.get('confirmation')
    if isinstance(confirmation, dict):
        q = confirmation.get('payload')
        return '确认单已准备，等待用户决定' + (('；' + quote_summary(q)) if isinstance(q, dict) else '')
    if name == 'query_order_tool':
        status = {'CONFIRMED': '已确认', 'CANCELLED': '已取消', 'DRAFT': '待确认'}.get(text(data.get('status')))
        if status:
            return '订单状态：' + status
    return '工具已返回结果'
