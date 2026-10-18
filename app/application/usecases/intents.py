"""请求执行输入与结果；HTTP、队列及图编排共用，不依赖具体执行器。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SubmitIntentInput:
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str
    raw_query: str
    confirmations: tuple[dict, ...] = ()
    message_id: str = ""
    run_id: str = ""


@dataclass(frozen=True)
class SubmitIntentOutput:
    shopping_session_id: str
    final_text: str
    error: str | None = None
    error_code: str | None = None
    status: str = "completed"
    stop_reason: str | None = None
    product_delivery_complete: bool = False
