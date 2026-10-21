"""从原生AG-UI事实读取评测观察，不重建旧总线事件协议。"""
import json


def decode_receipt(content):
    receipt = json.loads(content)
    if not isinstance(receipt, dict) or set(receipt) != {'status', 'data', 'failure'} or receipt['status'] not in {'success', 'error'}:
        raise ValueError('工具回执缺少当前status/data/failure')
    return receipt


def tool_observations(frames):
    calls = {}
    observations = []
    for index, frame in enumerate(frames):
        kind = frame.get('type')
        if kind == 'TOOL_CALL_START':
            calls[frame['toolCallId']] = {'name': frame['toolCallName'], 'arguments_text': '', 'index': index}
        elif kind == 'TOOL_CALL_ARGS':
            calls[frame['toolCallId']]['arguments_text'] += frame['delta']
        elif kind == 'TOOL_CALL_RESULT':
            call = calls[frame['toolCallId']]
            receipt = decode_receipt(frame['content'])
            raw = call['arguments_text']
            try:
                arguments = json.loads(raw)
            except ValueError:
                arguments = raw
            observations.append({'index': index, 'call_index': call['index'], 'call_id': frame['toolCallId'],
                'name': call['name'], 'arguments': arguments, **receipt})
    return observations


def tool_names(frames):
    return {f['toolCallName'] for f in frames if f.get('type') == 'TOOL_CALL_START'}


def presentations(frames, kind):
    seen = set()
    rows = []
    for index, frame in enumerate(frames):
        if frame.get('type') != 'STATE_SNAPSHOT':
            continue
        for item in frame['snapshot'].get('presentations', []):
            if item['id'] not in seen and item['kind'] == kind:
                rows.append((index, item['payload']))
                seen.add(item['id'])
    return rows


def final_results(frames):
    state = {}; messages = []; rows = []
    for frame in frames:
        kind = frame.get('type')
        if kind == 'RUN_STARTED': state = {}; messages = []
        elif kind == 'STATE_SNAPSHOT': state = frame['snapshot']
        elif kind == 'MESSAGES_SNAPSHOT': messages = frame['messages']
        elif kind in {'RUN_FINISHED', 'RUN_ERROR'}:
            rows.append({'text': frame['message'] if kind == 'RUN_ERROR' else next(
                (m['content'] for m in reversed(messages) if m['role'] == 'assistant'), ''),
                'status': 'failed' if kind == 'RUN_ERROR' else state.get('executionStatus', 'completed'),
                'stop_reason': frame.get('code') if kind == 'RUN_ERROR' else state.get('stopReason')})
    return rows


def usage_summaries(frames):
    return [f['value'] for f in frames if f.get('type') == 'CUSTOM' and f.get('name') == 'usage.summary']


def model_starts(frames):
    return [f for f in frames if f.get('type') == 'STEP_STARTED' and f.get('stepName', '').startswith('model:')]
