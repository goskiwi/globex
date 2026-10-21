"""验收脚本共用持久 AG-UI 通路，事件来自同一运行的公开 SSE。"""
from __future__ import annotations
import json
import uuid


def run_input(session_id, buyer_id, query):
    return {'threadId':session_id,'runId':uuid.uuid4().hex,'state':{},'tools':[],'context':[],
            'messages':[{'id':uuid.uuid4().hex,'role':'user','content':query}],
            'forwardedProps':{'buyerId':buyer_id,'locale':'zh-CN','currency':'CNY'}}


def read_stream(text):
    frames=[json.loads(line[5:].strip()) for line in text.splitlines() if line.startswith('data:')]
    if not frames or frames[-1]['type'] not in {'RUN_FINISHED','RUN_ERROR'}:
        raise ValueError('事件流未完成，不能作为成功验收')
    errors=[e for e in frames if e['type']=='RUN_ERROR']
    if errors:raise ValueError(errors[-1].get('message','运行失败'))
    messages=[e['messages'] for e in frames if e['type']=='MESSAGES_SNAPSHOT']
    final=next((m['content'] for m in reversed(messages[-1] if messages else []) if m['role']=='assistant'),'')
    events=[{'type':e['name'],'payload':e.get('value',{})} for e in frames if e['type']=='CUSTOM']
    return {'final_text':final},events,frames


async def submit_stream(client, base_url, session_id, buyer_id, query, *, timeout=600):
    response=await client.post(base_url.rstrip('/')+'/commerce/ag-ui/run',
        json=run_input(session_id,buyer_id,query),timeout=timeout)
    response.raise_for_status()
    result,events,frames=read_stream(response.text)
    return result,events,response
