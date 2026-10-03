"""对单个图任务的完整业务观察检测 A 重复及 AB 短周期；不持有跨会话历史。"""
from dataclasses import dataclass
import hashlib
import json

DEFAULT_REPEAT_THRESHOLD = 3
DEFAULT_WINDOW = 6
READ_TOOLS = frozenset({'product_search_tool', 'get_product_details', 'category_insight_tool',
                        'conversation_fact_lookup', 'web_search_tool'})
PROGRESS_HINT = '同一任务内的操作路径已经重复，完整业务结果未变化。'


def _business_result(tool, value):
    if isinstance(value, list):
        return [_business_result(tool, item) for item in value]
    if isinstance(value, dict):
        # 返回引用与观察时间不表示新业务信息；输入的引用、页码和游标不做此处理。
        return {k: _business_result(tool, v) for k, v in value.items()
                if k not in {'result_ref', 'observed_at'}}
    return value


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class LoopDetector:
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD
    window: int = DEFAULT_WINDOW

    def observation(self, tool, arguments, result, status, conditions):
        if tool not in READ_TOOLS or status not in {'success', 'error'}:
            return None
        return {'tool': tool, 'arguments': arguments,
                'action': _fingerprint([tool, arguments, conditions]),
                'result': _fingerprint([status, _business_result(tool, result)]), 'status': status}

    def advance(self, previous, observations):
        history = [*previous.get('history', []), *observations]
        count = previous.get('count', 0) + len(observations)
        history = history[-max(self.window, self.repeat_threshold, 4):]
        updated = {**previous, 'history': history, 'count': count, 'stop': previous.get('stop', False)}
        if not observations:
            return updated, None
        signatures = [(x['action'], x['result']) for x in history]
        for period, repeats in ((1, self.repeat_threshold), (2, 2)):
            width = period * repeats
            if len(signatures) < width:
                continue
            pattern = signatures[-period:]
            if period == 2 and pattern[0][0] == pattern[1][0]:
                continue
            if signatures[-width:] != pattern * repeats:
                continue
            # AB 与 BA 是同一个周期，不能因窗口起点移动而丢失已发出的纠偏提示。
            cycle = min(pattern[i:] + pattern[:i] for i in range(period))
            key = [list(x) for x in cycle]
            warned = previous.get('warned')
            if warned and warned['pattern'] == key:
                updated['stop'] = count - warned['count'] >= period
                return updated, None
            updated['warned'] = {'pattern': key, 'count': count}
            tried = [{'tool': x['tool'], 'arguments': x['arguments'], 'status': x['status']}
                     for x in history[-period:]]
            return updated, (PROGRESS_HINT + '\n已尝试：' + json.dumps(tried, ensure_ascii=False)
                + '\n请对照当前目标与未解决问题，采用有新依据的查询或交付已有结论和缺口。'
                  '调用成功但结果不变不等于服务失败；不要编造失败原因。继续相同周期将结束本轮自主探索。')
        updated.pop('warned', None)
        return updated, None
