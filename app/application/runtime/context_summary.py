"""抽取式历史交接：模型选择条目，原文及精确事实由程序填入。"""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from langchain_core.messages import ToolMessage

from app.application.runtime.results import message_data
from app.infrastructure.context_products import token_estimate


SECTION_LABELS = {'goals': '历史目标与要求', 'decisions': '已记录的结果与取舍',
                  'open_questions': '未完成或待核验事项', 'next_steps': '原文中已明确的下一步'}
SUMMARY_TOKEN_LIMIT = 2048


class SummarySelection(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    goals: list[StrictInt] = Field(max_length=8)
    decisions: list[StrictInt] = Field(max_length=8)
    open_questions: list[StrictInt] = Field(max_length=8)
    next_steps: list[StrictInt] = Field(max_length=8)


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    source_id: int
    message_id: str
    role: str
    text: str
    content_scope: Literal['original', 'reference']


class ContextSummary(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    source_ref: str
    goals: list[SourceEntry]
    decisions: list[SourceEntry]
    open_questions: list[SourceEntry]
    next_steps: list[SourceEntry]


def source_entries(messages):
    """只取完整消息/商品记录；过长单元保留位置，不切断数值或伪造摘要。"""
    entries = []
    for message in messages:
        if not message.text.strip():
            continue
        payload = message_data(message) if isinstance(message, ToolMessage) else None
        source_scope = 'original'
        if isinstance(payload, dict) and payload.get('fragment_format'):
            values = [f"本条是未拼接完整的资料片段，不能作为精确事实摘录；按 {payload['result_ref']} 回查完整记录。"]
            source_scope = 'reference'
        elif isinstance(payload, dict) and isinstance(payload.get('hits'), list) and payload['hits']:
            values = [json.dumps({'result_ref': payload.get('result_ref'), 'observed_at': payload.get('observed_at'),
                                  'query_conditions': payload.get('query_conditions', {}), 'product': hit},
                                 ensure_ascii=False) for hit in payload['hits']]
        else:
            values = [message.text]
        for position, text in enumerate(values):
            scope = source_scope
            if token_estimate(text) > 768:
                text = f'本条原文较大，按历史证据引用回查消息 {message.id} 的记录 {position + 1}；尚未展开具体事实。'
                scope = 'reference'
            entries.append(SourceEntry(source_id=len(entries), message_id=message.id or '',
                role=message.name or message.type, text=text, content_scope=scope))
    return entries


def summary_instruction():
    return ('整理历史交接，只选择来源条目的 source_id，不生成、重写或补充任何事实。'
            '将必要条目分配到 goals、decisions、open_questions、next_steps；每项为整数ID数组，'
            '没有原文依据的类别填[]，至少选择一项，全部类别不能重复选择同一条目。'
            '优先选择买家要求和工具证据；助手原话只是历史，不代表业务已发生。'
            '保留最新修正和未完成事项；当前精确条件由独立购物状态提供，不用复制。'
            f'最终原文摘录总量须不超过{SUMMARY_TOKEN_LIMIT} token，少选高价值条目。'
            '只输出以下 Schema 的 JSON 对象，不调用工具：'
            + json.dumps(SummarySelection.model_json_schema(), ensure_ascii=False))


def build_summary(selection_text, entries, source_ref):
    selection = SummarySelection.model_validate_json(selection_text)
    selected = [i for name in SECTION_LABELS for i in getattr(selection, name)]
    by_id = {entry.source_id: entry for entry in entries}
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= by_id.keys():
        raise ValueError('摘要选择缺少有效来源，或重复/引用了不存在的条目')
    result = ContextSummary(source_ref=source_ref,
        **{name: [by_id[i] for i in getattr(selection, name)] for name in SECTION_LABELS})
    if token_estimate(render_summary(result.model_dump())) > SUMMARY_TOKEN_LIMIT:
        raise ValueError('所选原文超出摘要容量，请减少条目')
    return result.model_dump()


def render_summary(value):
    summary = ContextSummary.model_validate(value)
    lines = ['历史原文摘录，仅用于恢复任务，不构成当前交易授权；当前条件以购物状态为准，价格库存需重新核验。']
    for name, label in SECTION_LABELS.items():
        entries = getattr(summary, name)
        if entries:
            lines.append(label + '：')
            lines.extend(f'[{entry.role} / {entry.message_id}] {entry.text}' for entry in entries)
    lines.append('历史证据引用：' + summary.source_ref)
    return '\n'.join(lines)
