"""请求执行输入与结果；HTTP、队列及图编排共用，不依赖具体执行器。"""
from dataclasses import dataclass
from typing import Literal, TypedDict


DELIVERY_KINDS = {"recommend_products":"recommendation", "compare_products":"comparison",
                  "show_product_details":"product_view", "present_research":"research",
                  "show_shopping_form":"clarification"}
PRODUCT_DELIVERIES = frozenset({"recommendation", "comparison", "product_view"})


class Delivery(TypedDict):
    kind: Literal["recommendation", "comparison", "product_view", "research", "clarification"]
    status: Literal["completed", "partial", "needs_input"]


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
    delivery: Delivery | None = None
