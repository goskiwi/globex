"""每次图执行的事实出口：调用身份来自原生消息，结果不依赖产生它的节点。"""
from contextvars import ContextVar
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from app.application.runtime.events import ToolEvent
from app.application.runtime.results import message_data, receipt_failure

_current: ContextVar['ExecutionFacts | None'] = ContextVar('execution_facts', default=None)


def call_identity(message, call_id):
    return f'{message.id}:{call_id}'


def execution_call(messages, call_id):
    message = next(m for m in reversed(messages) if isinstance(m, AIMessage)
                   and any(c['id'] == call_id for c in [*m.tool_calls, *m.invalid_tool_calls]))
    return message, next(c for c in [*message.tool_calls, *message.invalid_tool_calls] if c['id'] == call_id)


class ExecutionFacts:
    def __init__(self, observer):
        self.observer = observer
        self.calls = {}
        self.scopes = {}
        self.message_calls = {}
        self.started = set()
        self.results = set()
        self.begun = set()
        self.waiting = set()

    @staticmethod
    def current():
        return _current.get()

    def bind(self):
        return _current.set(self)

    @staticmethod
    def reset(token):
        _current.reset(token)

    def emit(self, kind, identifier, **fields):
        if self.observer:
            self.observer(ToolEvent(kind, identifier, **fields))

    def register(self, message, scope=(), *, announce=True):
        for call in [*message.tool_calls, *message.invalid_tool_calls]:
            identifier = call_identity(message, call['id'])
            self.calls[identifier] = call
            if scope is not None:
                self.scopes.setdefault(scope, {})[call['id']] = identifier
            if announce:
                self.announce(identifier)

    def announce(self, identifier):
        if identifier in self.started or identifier in self.results:
            return
        self.started.add(identifier)
        call = self.calls[identifier]
        arguments = call['args']
        self.emit('TOOL_CALL_START', identifier, tool_call_name=call['name'],
                  arguments=arguments if isinstance(arguments, dict) else None)
        self.emit('TOOL_CALL_DELTA', identifier, delta=json.dumps(arguments, ensure_ascii=False)
                  if isinstance(arguments, dict) else arguments or '')
        self.emit('TOOL_CALL_END', identifier)

    def begin(self, messages, call_id):
        message, call = execution_call(messages, call_id)
        self.register(message, None, announce=False)
        identifier = call_identity(message, call_id)
        self.announce(identifier)
        self.begun.add(identifier)
        self.emit('TOOL_RESULT_START', identifier, tool_call_name=call['name'])
        return identifier

    def result(self, identifier, receipt, data):
        if identifier in self.results:
            return
        self.announce(identifier)
        self.results.add(identifier)
        self.emit('TOOL_RESULT_END', identifier, tool_call_name=self.calls[identifier]['name'],
                  state=receipt.status, data=data, failure=receipt_failure(receipt))

    def observe(self, messages, scope=(), *, seed=False):
        for message in messages:
            if isinstance(message, AIMessage):
                self.register(message, scope, announce=not seed)
            elif isinstance(message, ToolMessage):
                identifier = self.message_calls.get(message.id) or self.scopes.get(scope, {}).get(message.tool_call_id)
                if identifier is not None and message.id is not None:
                    self.message_calls[message.id] = identifier
                if identifier is not None and seed:
                    self.results.add(identifier)
                elif identifier is not None:
                    data = (message.artifact or {}).get('event_data', (message.artifact or {}).get('data', message_data(message)))
                    self.result(identifier, message, data)

    def close_unfinished(self, reason):
        for identifier in sorted(self.started - self.results - self.waiting):
            call = self.calls[identifier]
            artifact = {'stop_reason': reason}
            if identifier not in self.begun:
                artifact['executed'] = False
            receipt = ToolMessage(tool_call_id=call['id'], name=call['name'], status='error',
                content='本次调用未取得完整结果；涉及写入请先核查状态，不直接重放。', artifact=artifact)
            self.result(identifier, receipt, receipt.content)

    def observe_update(self, update, scope=()):
        if isinstance(update, Command):
            self.observe_update(update.update, scope)
        elif isinstance(update, dict):
            self.observe(update.get('messages', []), scope)
        elif isinstance(update, (list, tuple)):
            for part in update:
                self.observe_update(part, scope)
