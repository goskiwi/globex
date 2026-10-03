"""比较同一会话中的业务进展；仅提示，不缓存结果或跳过真实核验。"""
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json

DEFAULT_REPEAT_THRESHOLD = 3
DEFAULT_WINDOW = 6
PROGRESS_HINT = (
    "近期对 {tool} 的相同参数调用已得到相同结果 {count} 次，没有新进展。"
    "请优先使用已有证据回答；确需补充时明确缺失字段，或按返回的下一页位置查询。"
    "正常翻页、不同商品/字段及结果变化不属于此提示；历史结果不能替代当前状态核验。"
)


def _business_result(tool, value):
    if isinstance(value, list):
        return [_business_result(tool, item) for item in value]
    if tool in {"product_search_tool", "get_product_details"} and isinstance(value, dict):
        # 只排除这次查询生成的元数据；商品更新时间、价格、库存、SKU 均保留。
        return {key: item for key, item in value.items() if key not in {"result_ref", "observed_at"}}
    return value


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass
class LoopDetector:
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD
    window: int = DEFAULT_WINDOW
    _observations: dict[str, deque] = field(default_factory=dict)

    def observe(self, session_id, tool_name, arguments, result, state):
        if state not in {"success", "error"}:
            return None
        history = self._observations.get(session_id)
        if history is None or history.maxlen != self.window:
            history = self._observations[session_id] = deque(history or (), maxlen=self.window)
        request_key = _fingerprint(arguments)
        result_key = state + ":" + _fingerprint(_business_result(tool_name, result))
        history.append((tool_name, request_key, result_key))
        count = 0
        for name, request, response in reversed(history):
            if name != tool_name or request != request_key:
                continue
            if response != result_key:
                break
            count += 1
        if count >= self.repeat_threshold:
            return PROGRESS_HINT.format(tool=tool_name, count=count)
        return None

    def reset(self, session_id):
        self._observations.pop(session_id, None)
