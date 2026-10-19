"""测试使用与生产一致的已接纳成果事件，不模拟旧的末尾交付开关。"""
from app.infrastructure.eventbus import TradeEvent,TradeEventBus
from app.application.runtime.middleware import BusinessToolMiddleware
from app.application.runtime.delivery import FinalDeliveryMiddleware


def presentation(kind,payload,*,identifier='presentation',run_id='r',status='completed'):
    return {'id':identifier,'runId':run_id,'kind':kind,'status':status,'payload':payload}


def accept(target,kind,payload,*,identifier='presentation',status='completed'):
    target.on_trade_event(TradeEvent(target.request.thread_id,'presentation.result',
        presentation(kind,payload,identifier=identifier,run_id=target.request.run_id,status=status),''))


def delivery_middlewares():
    return [BusinessToolMiddleware(None,TradeEventBus()),FinalDeliveryMiddleware()]
