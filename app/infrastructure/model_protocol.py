"""校验最终 HTTP 工具合同；不信任兼容网关会遵守 tool_choice。"""
from __future__ import annotations

class ModelProtocolViolation(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__('模型服务返回了不符合本次工具权限的响应，已停止执行。')


class ResponseContract:
    def __init__(self, request, observe):
        self.allowed = {t['function']['name'] for t in request.get('tools', [])
                        if t.get('type') == 'function'}
        choice = request.get('tool_choice')
        self.none = choice == 'none' or not self.allowed
        self.forced = choice.get('function', {}).get('name') if isinstance(choice, dict) else None
        self.required = choice == 'required' or bool(self.forced)
        self.observe = observe

    def validate(self, names):
        code = ('tools_forbidden' if self.none and names else
                'undeclared_tool' if any(n not in self.allowed for n in names) else
                'wrong_forced_tool' if self.forced and any(n != self.forced for n in names) else
                'required_tool_missing' if self.required and not names else None)
        self.observe(protocol_status=code or 'valid')
        if code:
            raise ModelProtocolViolation(code)
