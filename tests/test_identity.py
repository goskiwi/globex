"""本机签名身份、跨入口归属与持久 SSE 访问；只访问临时 SQLite。"""
from __future__ import annotations

from tests.sql_fixtures import session_store

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.session.ports.session_store import SessionNotFound, SessionOwnerMismatch
from app.infrastructure.eventbus import TradeEvent, TradeEventBus
from app.infrastructure.identity import IdentityError, IdentityPolicy
from app.infrastructure.persistence.sql.repositories import SqlSessionStore
from tests.test_confirmation_routes import order_request, decision
from tests.trade_test_helpers import confirmation_env  # noqa: F401

POLICY = IdentityPolicy(mode="hmac", secret="local-test-secret-never-used-in-production-123456")


def auth(buyer="buyer-1"):
    return {"Authorization": "Bearer " + POLICY.issue(buyer)}


def test_token_is_short_lived_scoped_and_secret_is_hidden():
    policy = replace(POLICY, clock=lambda: 1000)
    token = policy.issue("buyer-1", ttl_seconds=60)
    assert policy.verify(token) == "buyer-1"
    assert POLICY.secret not in repr(policy)
    with pytest.raises(IdentityError):
        replace(policy, clock=lambda: 1060).verify(token)
    with pytest.raises(IdentityError):
        replace(policy, clock=lambda: 900).verify(token)
    with pytest.raises(IdentityError):
        replace(policy, secret="wrong-secret-never-used-in-production-12345").verify(token)


@pytest.mark.parametrize("claims,header,algorithm", [
    ({"iss": "other"}, {}, "HS256"), ({"aud": "other"}, {}, "HS256"),
    ({"aud": ["globex-api", "other"]}, {}, "HS256"),
    ({"exp": 999}, {}, "HS256"), ({"exp": 9999999}, {}, "HS256"),
    ({"iat": True}, {}, "HS256"), ({"sub": ""}, {}, "HS256"),
    ({"admin": True}, {}, "HS256"), ({}, {"typ": "JWT"}, "HS256"),
    ({}, {"kid": "untrusted"}, "HS256"), ({}, {}, "HS384"),
    ({}, {}, "none"),
])
def test_invalid_token_claims_and_algorithms_are_rejected(claims, header, algorithm):
    policy = replace(POLICY, clock=lambda: 1000)
    payload = {"sub": "buyer-1", "iat": 1000, "exp": 1060, "iss": "globex-local", "aud": "globex-api", **claims}
    token = jwt.encode(payload, None if algorithm == "none" else policy.secret,
        algorithm=algorithm, headers={"typ": "globex-access+jwt", **header})
    with pytest.raises(IdentityError):
        policy.verify(token)


def test_strict_mode_configuration_fails_closed():
    with pytest.raises(ValueError):
        IdentityPolicy(mode="hmac", secret="too-short")
    with pytest.raises(ValueError):
        IdentityPolicy.from_settings(SimpleNamespace(identity_mode="hmac", identity_hmac_secret=POLICY.secret,
            session_owner_binding=False))
    with pytest.raises(ValueError):
        POLICY.issue("buyer", ttl_seconds=True)
    with pytest.raises(ValueError):
        IdentityPolicy().issue("buyer")


@pytest.fixture
async def secured(confirmation_env, monkeypatch,tmp_path):
    from app.presentation import server
    env = confirmation_env
    store = SqlSessionStore(env.engine)
    container = SimpleNamespace(
        confirmations=env.service, session_store=store, identity_policy=POLICY,
        settings=SimpleNamespace(llm_model="test", session_owner_binding=True, metrics_reader_buyers=(), metrics_scrape_token_file=""),
        bus=env.bus, backplane=None, startup=AsyncMock(), shutdown=AsyncMock(),
        task_queue=None, db_engine=env.engine, trade_db_engine=env.engine,
        cache=SimpleNamespace(enabled=False),
        orchestrator=SimpleNamespace(handle_intent=AsyncMock(return_value=SimpleNamespace(
            status="completed", stop_reason=None, error=None, presentations=(), shopping_session_id="session-1", final_text="结果"))),
        query_order=SimpleNamespace(execute=AsyncMock(return_value={"order_id": "order"})),
        cancel_order=SimpleNamespace(execute=AsyncMock(return_value={"confirmation": {}})),
    )
    from app.application.runtime.runs import RunService
    from app.infrastructure.ag_ui_journal import AGUIJournal
    container.runs=RunService(AGUIJournal(tmp_path/'runs.db'),container.orchestrator,session_store=store)
    container.startup=container.runs.startup
    container.shutdown=container.runs.shutdown
    monkeypatch.setattr(server, "build_container", AsyncMock(return_value=container))
    api = server.build_app()
    async with api.router.lifespan_context(api):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://local") as client:
            yield SimpleNamespace(client=client, store=store, container=container, env=env)


def intent(buyer="buyer-1", session="session-1"):
    return {"buyer_id": buyer, "shopping_session_id": session, "raw_query": "只读商品搜索"}


async def test_intent_binding_survives_instances_and_requires_matching_authenticated_owner(secured):
    c = secured.client
    assert (await c.get("/health")).status_code == 200
    assert (await c.post("/commerce/intents", json=intent())).status_code == 401
    assert (await c.post("/commerce/intents", json=intent(), params={"token": POLICY.issue("buyer-1")})).status_code == 401
    assert (await c.post("/commerce/intents", json=intent(), headers=auth())).status_code == 200
    assert (await c.post("/commerce/intents", json=intent("buyer-2"), headers=auth())).status_code == 403
    assert (await c.post("/commerce/intents", json=intent("buyer-2"), headers=auth("buyer-2"))).status_code == 403
    other = SqlSessionStore(secured.env.engine)
    with pytest.raises(SessionOwnerMismatch):
        await other.assert_owner("session-1", "buyer-2")
    secured.container.orchestrator.handle_intent.assert_awaited_once()


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/commerce/intents/async", intent("buyer-2")),
    ("POST", "/commerce/confirmations/orders", order_request(buyer_id="buyer-2")),
    ("POST", "/commerce/confirmations/id/resolve", {"buyer_id": "buyer-2", "session_id": "session-1", "approved": True, "snapshot_hash": "x"}),
    ("POST", "/commerce/orders/id/cancel", {"buyer_id": "buyer-2", "session_id": "session-1", "reason": "测试"}),
    ("GET", "/commerce/tasks/id?buyer_id=buyer-2", None),
    ("GET", "/commerce/orders/id?buyer_id=buyer-2", None),
    ("GET", "/commerce/confirmations?buyer_id=buyer-2&session_id=session-1", None),
    ("GET", "/commerce/confirmations/id?buyer_id=buyer-2&session_id=session-1", None),
])
async def test_every_existing_buyer_entry_rejects_identity_spoofing(secured, method, path, body):
    response = await secured.client.request(method, path, json=body, headers=auth())
    assert response.status_code == 403, response.text


async def test_confirmation_prepare_resolve_remains_usable_with_strict_identity(secured):
    response = await secured.client.post("/commerce/confirmations/orders", json=order_request(), headers=auth())
    assert response.status_code == 200, response.text
    confirmation = response.json()["confirmation"]
    assert (await secured.env.store.get_inventory())["P1001-S1"] == 50
    resolved = await secured.client.post(f"/commerce/confirmations/{confirmation['confirmation_id']}/resolve",
        json=decision(confirmation), headers=auth())
    assert resolved.status_code == 200 and resolved.json()["order"]["status"] == "CONFIRMED"


async def test_task_access_is_durable_and_bound_to_buyer_and_session(secured):
    store = secured.store
    await store.assert_owner("session-1", "buyer-1", create=True)
    await store.bind_task_owner("task-1", "session-1", "buyer-1")
    await store.bind_task_owner("task-1", "session-1", "buyer-1")
    another = SqlSessionStore(secured.env.engine)
    assert await another.assert_task_owner("task-1", "buyer-1") == "session-1"
    with pytest.raises(SessionOwnerMismatch):
        await another.bind_task_owner("task-1", "other-session", "buyer-1")
    with pytest.raises(SessionNotFound):
        await another.assert_task_owner("missing", "buyer-1")
    from ag_ui.core import RunAgentInput
    body=RunAgentInput.model_validate({'runId':'task-1','threadId':'session-1','state':{},
        'messages':[{'id':'u','role':'user','content':'只读商品搜索'}],'tools':[],'context':[],
        'forwardedProps':{'buyerId':'buyer-1'}})
    await secured.container.runs.start(body,'buyer-1')
    await secured.container.runs.wait('task-1','buyer-1')
    response = await secured.client.get("/commerce/tasks/task-1", params={"buyer_id": "buyer-1"}, headers=auth())
    assert response.status_code == 200 and response.json()["final_text"] == "结果"
    response = await secured.client.get("/commerce/tasks/task-1", params={"buyer_id": "buyer-2"}, headers=auth("buyer-2"))
    assert response.status_code == 403


async def test_retired_websocket_route_is_not_registered(secured):
    app=secured.client._transport.app
    assert not any(getattr(route,'path',None)=='/commerce/events' for route in app.routes)


@pytest.mark.parametrize("path", ["/internal/metrics", "/internal/metrics/summary"])
async def test_operational_metrics_are_disabled_by_default_and_never_in_public_health(secured, path):
    response = await secured.client.get(path, headers=auth())
    assert response.status_code == 404
    health = (await secured.client.get("/health")).json()
    assert "metrics" not in health and "alerts" not in health


@pytest.mark.parametrize("path", ["/internal/metrics", "/internal/metrics/summary"])
async def test_operational_metrics_require_allowlisted_signed_operator(secured, path):
    api = secured.client._transport.app
    api.state.metrics_reader_buyers = ("ops-reader",)
    assert (await secured.client.get(path)).status_code == 401
    assert (await secured.client.get(path, params={"token": POLICY.issue("ops-reader")})).status_code == 401
    assert (await secured.client.get(path, headers=auth())).status_code == 403
    result = await secured.client.get(path, headers=auth("ops-reader"))
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    if path.endswith("summary"):
        assert result.json()["metrics"]["scope"] == "process_local_business_turns"
    else:
        assert "version=0.0.4" in result.headers["content-type"]
    api.state.identity_policy = IdentityPolicy()
    assert (await secured.client.get(path, headers=auth("ops-reader"))).status_code == 503
