"""按原生消息归属关联工具调用；提供方 ID 仅在所属消息内标识调用。"""
from dataclasses import dataclass
from langchain_core.messages import AIMessage, ToolMessage


def call_identity(message, call_id):
    return f'{message.id}:{call_id}'


@dataclass
class ToolCallRecord:
    message: AIMessage
    call: dict
    index: int
    receipt: ToolMessage | None = None
    receipt_index: int | None = None

    @property
    def identifier(self):
        return call_identity(self.message, self.call['id'])


def tool_call_records(messages):
    records, pending, identities = [], {}, {}
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            for call in [*message.tool_calls, *message.invalid_tool_calls]:
                record = ToolCallRecord(message, call, index)
                records.append(record)
                pending.setdefault(call['id'], []).append(record)
                identities[record.identifier] = record
        elif isinstance(message, ToolMessage):
            record = identities.get((message.artifact or {}).get('execution_id'))
            if record is None:
                record = next((r for r in reversed(pending.get(message.tool_call_id, []))
                               if r.receipt is None), None)
            if record is not None:
                record.receipt, record.receipt_index = message, index
    return records


def execution_call(messages, call_id):
    record = next(r for r in reversed(tool_call_records(messages)) if r.call['id'] == call_id)
    return record.message, record.call
