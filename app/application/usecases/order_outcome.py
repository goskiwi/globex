"""订单任务的程序化说明与模型视图；不由模型改写金额或状态。"""
STATUSES = {'DRAFT':'草稿', 'CONFIRMED':'已确认', 'CANCELLED':'已取消'}


def order_guidance(outcome):
    order = outcome['order']
    amount = f"¥{order['total_amount_minor']/100:.2f}" if order['currency']=='CNY' else f"{order['total_amount_minor']/100:.2f} {order['currency']}"
    text = f"订单查看时状态：{STATUSES[order['status']]}，到手总额{amount}。"
    action = outcome['operation']
    if action['status']=='rejected':
        return text + '未准备取消确认：' + action['rejection']['message']
    if action['status']=='awaiting_confirmation':
        return text + '取消确认卡已准备，请在卡片上操作；确认前订单保持原状态。'
    return text + '本次只查看，未准备或执行取消。'


def order_model_receipt(outcome):
    return {'displayed':True, 'kind':'order', 'status':outcome['status'], 'observed_at':outcome['observed_at'],
        'order':{key:outcome['order'][key] for key in ('order_id','status','currency','total_amount_minor','created_at')},
        'operation':outcome['operation'], 'confirmation_required':outcome['confirmation_required'],
        'confirmation_id':outcome['confirmation']['confirmation_id'] if outcome['confirmation'] is not None else None}
