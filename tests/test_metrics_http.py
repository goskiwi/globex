"""保留原部署的指标采集权限和Worker HTTP生命周期，不调用业务模型。"""
import asyncio
import socket
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI,Request

from app.infrastructure.identity import IdentityPolicy
from app.presentation.identity import require_buyer
from app.presentation.metrics import configure_metrics_access,register_metrics_routes


def settings(tmp_path,port=0):
    token=tmp_path/'metrics-token';token.write_text('m'*40)
    return SimpleNamespace(metrics_scrape_token_file=str(token),metrics_reader_buyers=('ops',),
                           worker_metrics_port=port,worker_concurrency=1)


async def test_scrape_token_reads_metrics_but_never_authorizes_buyer_requests(tmp_path):
    app=FastAPI();policy=IdentityPolicy(mode='hmac',secret='s'*40)
    configure_metrics_access(app,settings(tmp_path),policy)
    class Queue:
        async def metrics(self):return {'depth':3,'lag':2,'pending':1,'dead_letters':0}
    register_metrics_routes(app,lambda:Queue())
    @app.get('/buyer')
    async def buyer(request:Request):return await require_buyer(request,'buyer')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        assert (await client.get('/internal/metrics')).status_code==401
        assert (await client.get('/internal/metrics',headers={'Authorization':'Bearer wrong'})).status_code==401
        scrape={'Authorization':'Bearer '+'m'*40}
        result=await client.get('/internal/metrics',headers=scrape)
        assert result.status_code==200 and 'globex_queue_depth 3' in result.text
        assert 'm'*40 not in result.text and result.headers['cache-control']=='no-store'
        assert (await client.get('/buyer',headers=scrape)).status_code==401
        assert (await client.get('/internal/metrics',headers={'Authorization':'Bearer '+policy.issue('buyer')})).status_code==403
        assert (await client.get('/internal/metrics/summary',headers={'Authorization':'Bearer '+policy.issue('ops')})).status_code==200


def test_invalid_scrape_credential_is_rejected(tmp_path):
    config=settings(tmp_path);(tmp_path/'metrics-token').write_text('short')
    with pytest.raises(ValueError):configure_metrics_access(FastAPI(),config,IdentityPolicy(mode='hmac',secret='s'*40))


async def test_disabled_metrics_remain_unavailable(tmp_path):
    config=settings(tmp_path);config.metrics_scrape_token_file='';config.metrics_reader_buyers=()
    app=FastAPI();configure_metrics_access(app,config,IdentityPolicy(mode='hmac',secret='s'*40));register_metrics_routes(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        assert (await client.get('/internal/metrics')).status_code==404


async def test_worker_exposes_authenticated_metrics_and_stops_with_consumer(tmp_path,monkeypatch):
    from app import worker
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    finished=asyncio.Event();closed=[]
    class Queue:
        async def consume(self,**kwargs):await finished.wait()
    class Container:
        task_queue=Queue();settings=None;identity_policy=IdentityPolicy(mode='hmac',secret='s'*40)
        async def startup(self):pass
        async def shutdown(self):closed.append(True)
    container=Container();container.settings=settings(tmp_path,port)
    async def build():return container
    monkeypatch.setattr(worker,'build_container',build)
    running=asyncio.create_task(worker.main())
    try:
        async with httpx.AsyncClient(timeout=.3) as client:
            async with asyncio.timeout(5):
                while True:
                    try:result=await client.get(f'http://127.0.0.1:{port}/internal/metrics')
                    except httpx.TransportError:await asyncio.sleep(.02);continue
                    break
            assert result.status_code==401
            result=await client.get(f'http://127.0.0.1:{port}/internal/metrics',headers={'Authorization':'Bearer '+'m'*40})
            assert result.status_code==200 and 'globex_request_duration_ms_count' in result.text
    finally:
        finished.set();await asyncio.wait_for(running,5)
    assert closed==[True]
