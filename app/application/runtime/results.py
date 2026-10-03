# -*- coding: utf-8 -*-
"""业务结果保存原生数据，文本仅在模型消息边界生成。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import json


class ToolResultState(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ToolErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BUSINESS_REJECTED = "business_rejected"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"


@dataclass
class ToolResult:
    data: Any
    state: ToolResultState = ToolResultState.SUCCESS
    error_code: ToolErrorCode | str | None = None
    model_data: Any = None
    event_data: Any = None
    error_reason: str | None = None

    def __post_init__(self):
        if (self.state == ToolResultState.ERROR) != (self.error_code is not None):
            raise ValueError("失败结果必须给出错误类别；成功结果不得带错误类别")
        self.error_code = ToolErrorCode(self.error_code) if self.error_code is not None else None
        if self.error_reason is not None and self.state != ToolResultState.ERROR:
            raise ValueError("成功结果不得带失败原因")

    @property
    def text(self):
        value = self.data if self.model_data is None else self.model_data
        return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)

    @property
    def ok(self) -> bool:
        return self.state == ToolResultState.SUCCESS


def message_data(message):
    """读取结构化模型投影；普通文本不冒充 JSON。"""
    artifact = getattr(message, 'artifact', None)
    if isinstance(artifact, dict) and 'data' in artifact:
        return artifact.get('model_data', artifact['data'])
    return getattr(message, 'content', None)


def input_failure(error, schema):
    """保留原生校验位置和类型；不传播输入值、上下文或异常文案。"""
    def field_label(path):
        node=schema; labels=[]
        for part in path:
            if '$ref' in node:
                node=schema.get('$defs',{}).get(node['$ref'].split('/')[-1],{})
            if isinstance(part,int):
                labels.append(f'第{part+1}项');node=node.get('items',{})
            else:
                node=node.get('properties',{}).get(part,{})
                labels.append(node.get('title',str(part)))
        return ' / '.join(labels)
    return {"code": "invalid_input", "executed": False, "issues": [
        {"path": list(item["loc"]), "type": item["type"], "label":field_label(item['loc'])}
        for item in error.errors(include_input=False, include_context=False, include_url=False)]}


def receipt_failure(message):
    """只读显式元数据；纯文字错误不推断为参数错误或业务拒绝。"""
    if message.status != 'error':
        return None
    artifact = message.artifact or {}
    if isinstance(artifact.get('failure'), dict):
        return artifact['failure']
    if artifact.get('stop_reason'):
        return {'code': 'stopped', 'reason': artifact['stop_reason']}
    code = artifact.get('error_code')
    result = {'code': code or 'unknown'}
    if type(artifact.get('executed')) is bool:
        result['executed'] = artifact['executed']
    if artifact.get('error_reason'):
        result['reason'] = artifact['error_reason']
    return result


def project_data(message, data, *, notices=None):
    artifact = dict(message.artifact or {})
    artifact.setdefault('data', data)
    if data == artifact['data']:
        artifact.pop('model_data',None)
    else:
        artifact['model_data'] = data
    if notices is not None: artifact['notices'] = notices
    shown = {'result':data, 'notices':artifact['notices']} if artifact.get('notices') else data
    content = shown if isinstance(shown,str) else json.dumps(shown,ensure_ascii=False)
    return message.model_copy(update={'content':content, 'artifact':artifact})


def tool_receipts(output):
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
    if isinstance(output, ToolMessage): return [output]
    if isinstance(output, Command) and isinstance(output.update,dict):
        return [m for m in output.update.get('messages',[]) if isinstance(m,ToolMessage)]
    return []


def replace_receipts(output, receipts):
    from dataclasses import replace
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
    if isinstance(output,ToolMessage): return receipts[0]
    if isinstance(output,Command):
        by_id={m.tool_call_id:m for m in receipts}
        messages=[by_id.get(m.tool_call_id,m) if isinstance(m,ToolMessage) else m
                  for m in output.update.get('messages',[])]
        return replace(output,update={**output.update,'messages':messages})
    return output
