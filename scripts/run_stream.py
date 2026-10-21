"""持久AG-UI流：先保留已收到事实，再由调用方判断是否成功。"""
from __future__ import annotations
from dataclasses import dataclass
import json
import uuid
import httpx
from ag_ui.core import Event
from pydantic import TypeAdapter

_EVENT = TypeAdapter(Event)
from scripts.eval.trace import final_results


def run_input(session_id, buyer_id, query):
    return {'threadId': session_id, 'runId': uuid.uuid4().hex, 'state': {}, 'tools': [], 'context': [],
        'messages': [{'id': uuid.uuid4().hex, 'role': 'user', 'content': query}],
        'forwardedProps': {'buyerId': buyer_id, 'locale': 'zh-CN', 'currency': 'CNY'}}


@dataclass
class RunStream:
    frames: list[dict]
    error: str | None
    final_text: str

    def require_success(self):
        if self.error:
            raise ValueError(self.error)


def read_stream(text):
    frames = []
    data = []
    error = None
    # SSE空行分帧；残缺JSON不能挤掉此前完整帧。
    for line in [*text.splitlines(), '']:
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif not line and data:
            try:
                frame = json.loads('\n'.join(data))
                _EVENT.validate_python(frame)
                frames.append(frame)
            except (ValueError, TypeError):
                error = '事件流包含不完整或无效JSON'
                break
            data = []
    failures = [f for f in frames if f.get('type') == 'RUN_ERROR']
    if failures:
        error = failures[-1].get('message', '运行失败')
    elif not frames or frames[-1].get('type') != 'RUN_FINISHED':
        error = error or '事件流未完成，不能作为成功验收'
    final = final_results(frames)
    return RunStream(frames, error, final[-1]['text'] if final else '')


async def submit_stream(client, base_url, session_id, buyer_id, query, *, timeout=600):
    chunks = []
    response = None
    connection_error = None
    try:
        async with client.stream('POST', base_url.rstrip('/') + '/commerce/ag-ui/run',
            json=run_input(session_id, buyer_id, query), timeout=timeout) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                chunks.append(chunk)
    except httpx.HTTPError as error:
        connection_error = type(error).__name__
    stream = read_stream(''.join(chunks))
    if connection_error and not (stream.frames and stream.frames[-1].get('type') in {'RUN_FINISHED','RUN_ERROR'}):
        stream.error = '运行连接中断：' + connection_error
    return stream, response
