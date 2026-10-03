"""原生消息的实验状态投影；保留既有增量格式及校验，执行仍由 LangGraph 负责。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from langchain_core.messages import HumanMessage, AIMessage

STATE_NAMES = {'shopping_state', 'shopping_state_delta'}
FORMAT = 'shopping-state-v2'


def projected_state(work):
    # 当前请求及本轮指代已经在完整买家消息中；精确工作状态不丢弃这些字段。
    return deepcopy({k: v for k, v in work.items()
                     if k not in {'latest_request', 'source_message_id', 'referenced'}})


def semantic_state(value):
    if isinstance(value, dict):
        return {k: semantic_state(v) for k, v in value.items() if k != 'message_id'}
    if isinstance(value, list):
        return [semantic_state(v) for v in value]
    return value


def state_hash(value):
    return hashlib.sha256(json.dumps(semantic_state(value), ensure_ascii=False,
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_message(payload):
    name = 'shopping_state' if payload['type'] == 'snapshot' else 'shopping_state_delta'
    return HumanMessage(name=name, content='<shopping-state>' + json.dumps(payload, ensure_ascii=False,
                   separators=(',', ':')) + '</shopping-state>', additional_kwargs={'shopping_state': payload})


def snapshot_message(work):
    state = projected_state(work)
    return state_message({'format': FORMAT, 'type': 'snapshot', 'state': state,
                          'revision': state_hash(state)[:16]})


def replay_state(messages):
    """只接受服务端写入的元数据链；缺基线/断链时要求新快照，不猜状态。"""
    current, revision = None, None
    count = 0
    for message in messages:
        if message.name not in STATE_NAMES:
            continue
        payload = message.additional_kwargs.get('shopping_state', {})
        if payload.get('format') != FORMAT:
            continue
        try:
            if payload['type'] == 'snapshot':
                current = deepcopy(payload['state'])
                count = 0
            elif payload['type'] == 'delta' and current is not None and payload['base'] == revision:
                for key in payload['remove']:
                    current.pop(key, None)
                current.update(deepcopy(payload['set']))
                filters = current.setdefault('filters', {})
                for key in payload['filters_remove']:
                    filters.pop(key, None)
                filters.update(deepcopy(payload['filters_set']))
                count += 1
            else:
                return None, None, 0
            revision = state_hash(current)[:16]
            if revision != payload['revision']:
                return None, None, 0
        except (KeyError, TypeError, AttributeError):
            return None, None, 0
    return current, revision, count


def next_state_message(messages, work):
    previous, revision, deltas = replay_state(messages)
    current = projected_state(work)
    if previous is None:
        return snapshot_message(work)
    if semantic_state(previous) == semantic_state(current):
        return None
    # 任务切换或增量过多时建立完整基线，避免让模型无限追补丁。
    if previous.get('goal') != current.get('goal') or deltas >= 8:
        return snapshot_message(work)
    updates = {k: deepcopy(v) for k, v in current.items() if k != 'filters'
               and (k not in previous or semantic_state(v) != semantic_state(previous[k]))}
    old_rules, new_rules = previous.get('filters', {}), current.get('filters', {})
    payload = {'format': FORMAT, 'type': 'delta', 'base': revision,
               'revision': state_hash(current)[:16], 'set': updates,
               'remove': sorted(previous.keys() - current.keys()),
               'filters_set': {k: deepcopy(v) for k, v in new_rules.items()
                    if k not in old_rules or semantic_state(v) != semantic_state(old_rules[k])},
               'filters_remove': sorted(old_rules.keys() - new_rules.keys())}
    delta = state_message(payload)
    snapshot = snapshot_message(work)
    # 小状态变化用全量反而更短时不强行使用补丁。
    return delta if len(delta.text) < len(snapshot.text) else snapshot


def project_state_messages(messages, work):
    current, _, _ = replay_state(messages)
    if current is not None and semantic_state(current) == semantic_state(projected_state(work)):
        return messages
    # 旧版本恢复/摘要切掉基线：仅重建请求投影，保留工具调用配对和原始消息。
    clean = [m for m in messages if m.name not in STATE_NAMES]
    source_id = work.get('source_message_id')
    position = next((i + 1 for i, m in enumerate(clean) if m.id == source_id), None)
    if position is None:
        # 找不到买家锚点时，放在系统及摘要之后、工具对之前。
        position = next((i for i, m in enumerate(clean) if isinstance(m, AIMessage)), len(clean))
    return [*clean[:position], snapshot_message(work), *clean[position:]]
