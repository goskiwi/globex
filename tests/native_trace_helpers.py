"""构造当前AG-UI事实，用于评测断言的正例及故障注入。"""
import json
import uuid


def tool_frames(name, data=None, *, status='success', arguments=None, failure=None):
    identifier=uuid.uuid4().hex
    return [{'type':'TOOL_CALL_START','toolCallId':identifier,'toolCallName':name},
            {'type':'TOOL_CALL_ARGS','toolCallId':identifier,'delta':json.dumps(arguments or {},ensure_ascii=False)},
            {'type':'TOOL_CALL_END','toolCallId':identifier},
            {'type':'TOOL_CALL_RESULT','toolCallId':identifier,'messageId':identifier+'-result',
             'role':'tool','content':json.dumps({'status':status,'data':data,'failure':failure},ensure_ascii=False)}]


def presentation_frame(kind, payload):
    return {'type':'STATE_SNAPSHOT','snapshot':{'presentations':[{'id':uuid.uuid4().hex,
            'runId':'test-run','kind':kind,'status':'completed','payload':payload}]}}


def final_frames(text='',status='completed',stop_reason=None):
    return [{'type':'MESSAGES_SNAPSHOT','messages':[{'id':uuid.uuid4().hex,'role':'assistant','content':text}]},
            {'type':'STATE_SNAPSHOT','snapshot':{'executionStatus':status,'stopReason':stop_reason}},
            {'type':'RUN_FINISHED','threadId':'test-thread','runId':'test-run'}]
