# -*- coding: utf-8 -*-
"""完整工具输出的基本结构检查；交易前置条件仍由业务工具核验。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# 工具名 → 期望返回结构里必须存在的关键字段
# （逐个对过真实工具实现：product_search 回 hits/recall_strategy，
#   category_insight 回 insights，订单三工具回 Order.snapshot()）
TOOL_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "product_search_tool": ("hits", "recall_strategy"),
    "get_product_details": ("hits", "existence_checked", "missing_identifiers"),
    "category_insight_tool": ("insights",),
    "create_order_tool": ("confirmation_required", "confirmation"),
    "query_order_tool": ("order_id", "status"),
    "cancel_order_tool": ("confirmation_required", "confirmation"),
}

@dataclass
class AssertionOutcome:
    """一次断言的结论。"""

    failures: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reject_reason: Optional[str] = None

    @property
    def rejected(self) -> bool:
        return self.reject_reason is not None


def check_schema(tool_name: str, tool_result: Any) -> AssertionOutcome:
    """Schema 断言：返回是否是含关键字段的结构化对象。"""
    outcome = AssertionOutcome()
    required = TOOL_REQUIRED_FIELDS.get(tool_name)
    if not required:
        return outcome  # 不在检查范围

    data: Any = tool_result

    if not isinstance(data, dict):
        outcome.failures.append(
            {"type": "schema", "tool": tool_name, "reason": "工具返回不是 JSON 对象"},
        )
        return outcome

    missing = [key for key in required if key not in data]
    if missing:
        outcome.failures.append(
            {
                "type": "schema",
                "tool": tool_name,
                "reason": f"缺少必需字段：{', '.join(missing)}",
            },
        )
    if tool_name in {'product_search_tool','get_product_details'} and 'hits' in data and not isinstance(data['hits'],list):
        outcome.failures.append({'type':'schema','tool':tool_name,'reason':'hits 必须是列表'})
    return outcome
