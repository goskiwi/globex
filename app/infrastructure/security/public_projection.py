"""所有公开业务序列化共享既有输出审核；内部记录与授权字段保持原样。"""
from copy import deepcopy
import json

from app.infrastructure.security.output_guard import audit_output

# 这些是既有 DTO 中的协议/授权字段，不是可改写的说明文字。
_PROTOCOL_FIELDS = frozenset('''id runId threadId messageId userMessageId toolCallId toolCallName
interruptId action_id interrupt_id form_id confirmationId taskId task_ids source_ref source_message_ids shopping_session_id session_id buyer_id product_id sku_id default_sku_id
canonical_product_id external_product_id confirmation_id operation_id order_id task_id
task_ids product_ids sku_ids requested_sku_ids excluded_product_ids excluded_sku_ids source_message_id source_run_id tool_call_id tool_call_name tool type role status
kind action decision code error_code currency source_currency target_currency ship_to country
snapshot_hash content_hash contentHash version result_ref occurred_at created_at updated_at
expires_at resolved_at timestamp run_id request_id checkpoint_id'''.split())


class PublicProjection:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def business(self, value):
        if not self.enabled:
            return deepcopy(value)
        if isinstance(value,dict):
            return {key:deepcopy(item) if key in _PROTOCOL_FIELDS else self.business(item)
                    for key,item in value.items()}
        if isinstance(value,list):
            return [self.business(item) for item in value]
        return audit_output(value)[1]

    def messages(self, messages):
        return [{**deepcopy(m), **({'content':self.business(m['content'])}
                if m.get('role')=='assistant' and 'content' in m else {})} for m in messages]

    def encoded_business(self, value):
        if not self.enabled:return deepcopy(value)
        try:
            parsed=json.loads(value)
        except (ValueError,TypeError):
            return self.business(value)
        return json.dumps(self.business(parsed),ensure_ascii=False)

    def trade(self, event_type, payload):
        result=self.business(payload)
        if event_type=='tool.lifecycle' and payload.get('type') in {'TOOL_CALL_DELTA','TOOL_RESULT_TEXT_DELTA'}:
            result['delta']=self.encoded_business(payload.get('delta',''))
        return result

    def event(self, event):
        result=deepcopy(event)
        kind=event.get('type')
        if kind=='MESSAGES_SNAPSHOT':result['messages']=self.messages(event['messages'])
        elif kind=='STATE_SNAPSHOT':result['snapshot']=self.business(event['snapshot'])
        elif kind=='STATE_DELTA':
            result['delta']=[{**op,**({'value':self.business(op['value'])} if 'value' in op else {})} for op in event['delta']]
        elif kind=='TOOL_CALL_RESULT':result['content']=self.encoded_business(event['content'])
        elif kind=='TOOL_CALL_ARGS':result['delta']=self.encoded_business(event['delta'])
        elif kind=='TEXT_MESSAGE_CONTENT':result['delta']=self.business(event['delta'])
        elif kind=='RUN_ERROR':result['message']=self.business(event['message'])
        elif kind=='CUSTOM':result['value']=self.business(event.get('value'))
        return result

    def run(self, value):
        result=deepcopy(value)
        result['messages']=self.messages(result.get('messages',[]))
        result['state']=self.business(result.get('state',{}))
        if result.get('result') is not None:result['result']=self.business(result['result'])
        # 可信执行输入仅由 runtime 消费，公开响应不包含私有执行调度信息。
        result.pop('input',None)
        return result

    def session(self, value):
        result=self.business({k:v for k,v in value.items() if k not in {'messages','run'}})
        result['messages']=self.messages(value.get('messages',[]))
        if value.get('run') is not None:result['run']=self.run(value['run'])
        return result
