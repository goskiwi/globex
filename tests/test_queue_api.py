"""REST与页面共用持久运行：两种调度模式下身份、结果与取消语义一致。"""
import asyncio
from types import SimpleNamespace
import httpx
import pytest
from app.application.runtime.runs import RunService
from app.application.usecases.intents import SubmitIntentOutput
from app.infrastructure.ag_ui_journal import AGUIJournal
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.identity import IdentityPolicy
from app.infrastructure.persistence.sql.repositories import SqlSessionStore,create_engine,bootstrap_schema
from app.infrastructure.queue.redis_stream_queue import RedisStreamTaskQueue,_STREAM
from app.presentation import server
from app.worker import execute_intent_task
from tests.test_queue_reliability import real_redis, isolated_redis_url

@pytest.fixture(params=[False,True],ids=['inline','queued'])
async def container(request,real_redis,tmp_path):
    engine=create_engine('sqlite+aiosqlite:///'+str(tmp_path/'sessions.db'))
    await bootstrap_schema(engine)
    store=SqlSessionStore(engine);bus=TradeEventBus();calls=[];stop=asyncio.Event();dispatch=asyncio.Event();dispatch.set()
    policy=IdentityPolicy(mode='hmac',secret='isolated-run-service-test-secret-not-production')
    async def execute(intent,**kwargs):
        assert kwargs['persistence_guard']()
        calls.append(intent)
        return SubmitIntentOutput(intent.shopping_session_id,'结果：'+intent.raw_query,status='completed')
    queue=RedisStreamTaskQueue(real_redis) if request.param else None
    runs=RunService(AGUIJournal(tmp_path/'runs.db'),SimpleNamespace(handle_intent=execute),task_queue=queue,session_store=store)
    worker=None
    async def handler(task):
        await dispatch.wait()
        return await execute_intent_task(task,runs,bus)
    async def startup():
        nonlocal worker
        await runs.startup()
        if queue is not None:
            worker=asyncio.create_task(queue.consume('run-test',handler,stop.is_set,block_ms=10,reclaim_idle_ms=100,lease_ms=1000))
    async def shutdown():
        dispatch.set();stop.set()
        if worker is not None:await worker
        await runs.shutdown();await engine.dispose()
    return SimpleNamespace(runs=runs,orchestrator=runs.orchestrator,task_queue=queue,bus=bus,backplane=None,
        settings=SimpleNamespace(queue_wait_seconds=3,data_dir=tmp_path),session_store=store,identity_policy=policy,
        startup=startup,shutdown=shutdown,calls=calls,dispatch=dispatch)

@pytest.fixture
async def api_client(container,monkeypatch):
    async def build():return container
    monkeypatch.setattr(server,'build_container',build)
    monkeypatch.setattr(server,'build_container_origins',lambda:[])
    app=server.build_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',
            headers={'Authorization':'Bearer '+container.identity_policy.issue('api-buyer')}) as client:yield client

def body(**changes):
    return {'shopping_session_id':'api-session','buyer_id':'api-buyer','request_id':'request-1','raw_query':'找背包',**changes}

@pytest.mark.parametrize('endpoint',['/commerce/intents','/commerce/intents/async'])
async def test_http_request_identity_is_independent_of_dispatch(api_client,container,endpoint):
    first=await api_client.post(endpoint,json=body());second=await api_client.post(endpoint,json=body())
    assert first.status_code==second.status_code==200
    if endpoint.endswith('async'):
        assert first.json()['task_id']==second.json()['task_id']
        run=await container.runs.wait(first.json()['task_id'],'api-buyer',3)
        assert run['result']['text']=='结果：找背包'
    else:assert first.json()==second.json()
    assert len(container.calls)==1
    conflict=await api_client.post(endpoint,json=body(raw_query='改找耳机'))
    assert conflict.status_code==409 and len(container.calls)==1

async def test_concurrent_new_session_retries_admit_one_run(api_client,container):
    responses=await asyncio.gather(*(api_client.post('/commerce/intents/async',json=body(shopping_session_id=None)) for _ in range(8)))
    assert all(r.status_code==200 for r in responses)
    assert len({r.json()['task_id'] for r in responses})==len({r.json()['shopping_session_id'] for r in responses})==1
    run=await container.runs.wait(responses[0].json()['task_id'],'api-buyer',3)
    assert run['status']=='completed' and len(container.calls)==1

async def test_request_keys_keep_sessions_and_buyers_separate(api_client,container):
    first=await api_client.post('/commerce/intents',json=body())
    second=await api_client.post('/commerce/intents',json=body(shopping_session_id='other-session'))
    other=await api_client.post('/commerce/intents/async',json=body(buyer_id='other-buyer',shopping_session_id='other-buyer-session'),
        headers={'Authorization':'Bearer '+container.identity_policy.issue('other-buyer')})
    assert first.status_code==second.status_code==other.status_code==200
    run=await container.runs.wait(other.json()['task_id'],'other-buyer',3)
    assert run['result']['text']=='结果：找背包'
    forbidden=await api_client.get('/commerce/tasks/'+other.json()['task_id'],params={'buyer_id':'api-buyer'})
    assert forbidden.status_code==403 and len(container.calls)==3

async def test_same_query_with_new_request_identity_is_a_new_run(api_client,container):
    await api_client.post('/commerce/intents',json=body())
    await api_client.post('/commerce/intents',json=body(request_id='request-2'))
    assert len(container.calls)==2

async def test_page_and_rest_share_queue_and_persistent_result(api_client,container,real_redis):
    request={'threadId':'page-session','runId':'page-run','state':{},'tools':[],'context':[],
             'messages':[{'id':'page-user','role':'user','content':'页面需求'}],'forwardedProps':{'buyerId':'api-buyer'}}
    response=await api_client.post('/commerce/ag-ui/run',json=request)
    assert response.status_code==200 and 'RUN_FINISHED' in response.text
    assert len(container.calls)==1 and container.calls[0].run_id=='page-run'
    if container.task_queue is not None:assert await real_redis.xlen(_STREAM)==1
    saved=await api_client.get('/commerce/tasks/page-run',params={'buyer_id':'api-buyer'})
    assert saved.status_code==200 and saved.json()['state']=='done' and saved.json()['final_text']=='结果：页面需求'

async def test_cancel_queued_run_does_not_execute(api_client,container):
    if container.task_queue is None:pytest.skip('排队阶段仅队列模式存在')
    container.dispatch.clear()
    response=await api_client.post('/commerce/intents/async',json=body())
    identifier=response.json()['task_id']
    stopped=await api_client.post('/commerce/ag-ui/runs/'+identifier+'/cancel',params={'buyer_id':'api-buyer'})
    assert stopped.status_code==200 and stopped.json()['status']=='stopped'
    container.dispatch.set();await asyncio.sleep(.1)
    assert container.calls==[]
