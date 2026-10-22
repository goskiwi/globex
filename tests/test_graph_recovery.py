"""统一配对、异常收尾和新进程恢复：验证实际持久图与模型HTTP请求。"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import httpx
import pytest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from app.application.runtime.context import summary_boundary
from app.application.runtime.tool_history import tool_call_records
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_execution_facts import request
from tests.test_langgraph_runtime import _container
from tests.test_order_task import create
from tests.native_model_helpers import client_model,completion


def assert_protocol(messages):
    pending=[]
    for message in messages:
        if message['role'] in {'user','assistant'}:
            assert not pending, '上一条调用尚未取得对应回执'
        if message['role']=='assistant':
            pending.extend(c['id'] for c in message.get('tool_calls',[]))
        if message['role']=='tool':
            assert message['tool_call_id'] in pending
            pending.remove(message['tool_call_id'])
    assert not pending


@pytest.mark.parametrize('reuse',[False,True])
def test_summary_retains_call_and_result_together(reuse):
    second='same' if reuse else 'different'
    messages=[HumanMessage(content='一'),AIMessage(content='',id='a1',tool_calls=[{'id':'same','name':'read','args':{}}]),
        HumanMessage(content='批准'),ToolMessage(content='一结果',tool_call_id='same'),HumanMessage(content='三'),
        AIMessage(content='',id='a2',tool_calls=[{'id':second,'name':'read','args':{}}]),
        ToolMessage(content='三结果',tool_call_id=second),HumanMessage(content='四')]
    assert summary_boundary(messages,[0,2,4,7],set())==0
    assert len({r.identifier for r in tool_call_records(messages)})==2


class ReusedIDs(ScriptedModel):
    def _generate(self,*args,**kwargs):
        result=super()._generate(*args,**kwargs)
        for item in result.generations[0].message.tool_calls:item['id']=item['id'].rsplit('-round-',1)[0]
        return result


async def next_round(c,path,monkeypatch,*,expire=False):
    bodies=[]
    def handler(req):
        body=json.loads(req.content);assert_protocol(body['messages']);bodies.append(body)
        return httpx.Response(200,json=completion())
    model=await client_model(path,handler)
    c=await _container(path,monkeypatch,model)
    try:
        if expire:
            # 模拟真实租约等待期已过；期限内拒绝新运行是应保留的行为。
            async with c.runs.journal._db(True) as db:
                await db.execute("UPDATE agui_runs SET lease_until=0 WHERE run_id='crash'")
        await c.runs.startup();await c.runs.start(request('recovered'),'buyer')
        run=await c.runs.wait('recovered','buyer',10)
        assert run['status']=='completed' and len(bodies)==1
        session=c.orchestrator._sessions._agents['facts-session'];snapshot=await session.graph.aget_state(session.config)
        assert not snapshot.next and all(r.receipt is not None for r in tool_call_records(snapshot.values['messages']))
        cards=(await c.confirmations.restore('buyer','facts-session'))['confirmations']
        assert len([x for x in cards if x['action']=='cancel'])==1
    finally:
        await c.shutdown();await model.aclose()


@pytest.mark.parametrize('origin',['node','user'])
@pytest.mark.parametrize('reuse',[False,True])
async def test_committed_tool_cancellation_closes_before_restart(tmp_path,monkeypatch,origin,reuse):
    model=(ReusedIDs if reuse else ScriptedModel)(responses=[])
    c=await _container(tmp_path,monkeypatch,model);held=None
    try:
        order=await create(SimpleNamespace(service=c.confirmations,store=c.trade_store),'facts-session')
        model.responses.append(call('handle_order',{'order_id':order['order_id']},turn_action='finish'))
        await c.runs.startup();await c.runs.start(request('view'),'buyer');await c.runs.wait('view','buyer',10)
        original=c.handle_order.execute;committed=asyncio.Event()
        async def interrupted(*a,**kw):
            await original(*a,**kw)
            if origin=='user':committed.set();await asyncio.Event().wait()
            raise asyncio.CancelledError()
        monkeypatch.setattr(c.handle_order,'execute',interrupted)
        model.responses.append(call('handle_order',{'order_id':order['order_id'],'cancellation':{'reason':'合成中断'}},turn_action='finish'))
        await c.runs.start(request('interrupted'),'buyer')
        if origin=='user':
            await asyncio.wait_for(committed.wait(),5);held=c.orchestrator._sessions._agents['facts-session']
            await c.runs.cancel('interrupted','buyer')
        run=await c.runs.wait('interrupted','buyer',10)
        assert run['status']==('stopped' if origin=='user' else 'error')
        session=c.orchestrator._sessions._agents.get('facts-session') or held
        snapshot=await session.graph.aget_state(session.config)
        records=tool_call_records(snapshot.values['messages'])
        assert len(records)==2 and not snapshot.next
        assert records[0].receipt.status=='success' and records[1].receipt.status=='error'
        assert (records[1].receipt.artifact or {}).get('executed') is not False
    finally:await c.shutdown()
    await next_round(c,tmp_path,monkeypatch)


async def test_failed_close_is_persistence_failure_and_later_recovers(tmp_path,monkeypatch):
    model=ScriptedModel(responses=[]);c=await _container(tmp_path,monkeypatch,model)
    import app.application.runtime.graph_execution as module
    try:
        order=await create(SimpleNamespace(service=c.confirmations,store=c.trade_store),'facts-session')
        original=c.handle_order.execute
        async def interrupted(*a,**kw):await original(*a,**kw);raise asyncio.CancelledError()
        monkeypatch.setattr(c.handle_order,'execute',interrupted)
        model.responses.append(call('handle_order',{'order_id':order['order_id'],'cancellation':{'reason':'合成保存故障'}},turn_action='finish'))
        async def failed_save(*a,**kw):raise OSError('合成持久化失败')
        with monkeypatch.context() as fault:
            fault.setattr(module,'close_graph',failed_save)
            await c.runs.startup();await c.runs.start(request('save-failed'),'buyer')
            run=await c.runs.wait('save-failed','buyer',10)
            assert run['state']['errorCode']=='PERSISTENCE_FAILED'
    finally:await c.shutdown()
    await next_round(c,tmp_path,monkeypatch)


async def test_process_exit_after_commit_is_reconciled_without_replaying(tmp_path,monkeypatch):
    # 独立进程直接退出，确保不依赖 finally、内存 ExecutionFacts 或优雅关停。
    script=tmp_path/'crash.py'
    script.write_text('''import asyncio,os,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.test_langgraph_runtime import _container
from tests.test_agent_handoff import ScriptedModel,call
from tests.test_execution_facts import request
from tests.test_order_task import create
async def main():
 p=Path(sys.argv[1]);m=ScriptedModel(responses=[]);c=await _container(p,pytest.MonkeyPatch(),m)
 o=await create(SimpleNamespace(service=c.confirmations,store=c.trade_store),'facts-session')
 original=c.handle_order.execute
 async def crash(*a,**kw):
  await original(*a,**kw)
  os._exit(37)
 c.handle_order.execute=crash
 m.responses.append(call('handle_order',{'order_id':o['order_id'],'cancellation':{'reason':'合成进程退出'}},turn_action='finish'))
 await c.runs.startup();await c.runs.start(request('crash'),'buyer');await asyncio.Event().wait()
asyncio.run(main())
''')
    process=await asyncio.create_subprocess_exec(sys.executable,str(script),str(tmp_path),
        env={**os.environ,'PYTHONPATH':str(Path.cwd()),'PROMPT_PIN_VERSION':''},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert await asyncio.wait_for(process.wait(),20)==37
    await next_round(None,tmp_path,monkeypatch,expire=True)
