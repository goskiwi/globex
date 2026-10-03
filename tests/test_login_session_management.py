"""签名登录、对话改名和完整删除；临时SQLite，无真实买家数据。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,MessagesState,START,END

from app.infrastructure.identity import IdentityPolicy
from app.infrastructure.ag_ui_journal import AGUIJournal,JournalNotFound,JournalConflict
from app.infrastructure.persistence.sql.repositories import create_engine,bootstrap_schema,SqlSessionStore,SqlConversationStore
from app.infrastructure.persistence.graph_checkpointer import FencedSqliteSaver
from app.infrastructure.persistence.context_evidence import ContextEvidenceStore
from app.infrastructure.shopping_forms import ShoppingFormStore
from app.infrastructure.context import ShoppingContext,ShoppingContextSnapshot
from app.domain.session.ports.conversation_store import ConversationTurn
from app.domain.session.ports.session_store import SessionNotFound,StaleSessionWrite
from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.usecases.session_management import SessionManagement
from app.presentation.auth import register_auth_routes
from app.presentation.session_management import register_session_management_routes

POLICY=IdentityPolicy(mode='hmac',secret='isolated-test-key-never-used-by-running-application')


@pytest.fixture
async def environment(tmp_path):
    engine=create_engine(f'sqlite+aiosqlite:///{tmp_path/"main.db"}')
    await bootstrap_schema(engine)
    store=SqlSessionStore(engine);conversations=SqlConversationStore(engine)
    journal=AGUIJournal(tmp_path/'journal.db');evidence=ContextEvidenceStore(tmp_path/'evidence.db')
    forms=ShoppingFormStore(tmp_path/'forms.db')
    async with FencedSqliteSaver.from_conn_string(str(tmp_path/'graph.db')) as saver:
        saver.session_store=store;await saver.setup()
        graph=StateGraph(MessagesState);graph.add_node('echo',lambda state:{});graph.add_edge(START,'echo');graph.add_edge('echo',END)
        compiled=graph.compile(checkpointer=saver)
        runner=MainAgentOrchestrator.__new__(MainAgentOrchestrator)
        runner._session_locks={};runner._session_lease_factory=None
        runner._sessions=SimpleNamespace(invalidate=AsyncMock())
        service=SessionManagement(runner,journal,store,conversations,evidence,forms,saver)
        api=FastAPI();api.state.identity_policy=POLICY
        register_auth_routes(api,lambda:SimpleNamespace(demo_login_password='123'))
        register_session_management_routes(api,lambda:service)
        async def seed(buyer,sid):
            claim=await store.claim(sid,buyer_id=buyer)
            scope=ShoppingContext.set(ShoppingContextSnapshot(sid,buyer,'zh-CN','CNY',session_fence=claim.fence))
            try:
                await compiled.ainvoke({'messages':[HumanMessage(content='合成对话正文')]},{'configurable':{'thread_id':sid}})
                await store.save_claim(claim,'{"runtime":"langgraph"}')
                await conversations.touch_session(sid,buyer,'zh-CN','CNY')
                await conversations.append_turn(ConversationTurn(sid,buyer,'buyer','合成对话正文'))
                ref=await evidence.save(buyer,sid,'products',{'hits':[{'product_id':'P1003'}]})
                await forms.create_clarification(buyer,sid,'合成问题',[{'id':'purpose','type':'text','label':'用途'}])
                body={'threadId':sid,'runId':'run-'+sid,'messages':[{'id':'u','role':'user','content':'原始标题'}],
                    'state':{},'tools':[],'context':[],'forwardedProps':{'buyerId':buyer}}
                await journal.reserve(body,buyer,'owner')
                await journal.append('run-'+sid,'owner',[{'type':'RUN_STARTED','threadId':sid,'runId':'run-'+sid},
                    {'type':'RUN_FINISHED','threadId':sid,'runId':'run-'+sid}])
                return claim,ref,body
            finally:ShoppingContext.reset(scope)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api),base_url='http://test') as client:
            yield SimpleNamespace(client=client,seed=seed,store=store,journal=journal,evidence=evidence,
                forms=forms,saver=saver,compiled=compiled,service=service,conversations=conversations)
    await engine.dispose()


async def login(client,buyer):
    response=await client.post('/commerce/auth/login',json={'account':buyer,'password':'123'})
    assert response.status_code==200
    return {'Authorization':'Bearer '+response.json()['accessToken']}


async def test_both_accounts_login_wrong_password_and_verified_subject(environment):
    c=environment.client
    for buyer in ('kkqq','root'):
        headers=await login(c,buyer)
        assert (await c.get('/commerce/auth/me',headers=headers)).json()=={'buyerId':buyer}
    assert (await c.get('/commerce/auth/me')).status_code==401
    assert (await c.post('/commerce/auth/login',json={'account':'kkqq','password':'wrong'})).status_code==401
    assert (await c.post('/commerce/auth/login',json={'account':'other','password':'123'})).status_code==422


async def test_rename_and_delete_are_owned_and_remove_all_dialogue_stores(environment):
    e=environment;old,ref,_=await e.seed('kkqq','dialogue-a');await e.seed('root','dialogue-b')
    a,b=await login(e.client,'kkqq'),await login(e.client,'root')
    url='/commerce/ag-ui/sessions/dialogue-a'
    assert (await e.client.patch(url,params={'buyer_id':'kkqq'},json={'title':'新标题'},headers=b)).status_code==403
    assert (await e.client.delete(url,params={'buyer_id':'root'},headers=b)).status_code==403
    assert (await e.client.patch(url,params={'buyer_id':'kkqq'},json={'title':'新标题'},headers=a)).status_code==200
    assert (await AGUIJournal(e.journal.path).sessions('kkqq'))[0]['title']=='新标题'
    assert (await e.client.delete(url,params={'buyer_id':'kkqq'},headers=a)).status_code==200
    assert await e.journal.sessions('kkqq')==[]
    assert len(await e.journal.sessions('root'))==1
    with pytest.raises(JournalNotFound):await e.journal.session('dialogue-a','kkqq')
    assert await e.saver.aget_tuple({'configurable':{'thread_id':'dialogue-a'}}) is None
    assert await e.evidence.get('kkqq','dialogue-a',ref) is None
    assert await e.forms.latest('kkqq','dialogue-a') is None
    assert await e.conversations.list_turns('dialogue-a')==[]
    assert await e.store.load('dialogue-a') is None
    await e.store.assert_owner('dialogue-a','kkqq',create=False) # 独立订单仍能校验归属。
    with pytest.raises(SessionNotFound):await e.store.claim('dialogue-a',buyer_id='kkqq')
    with pytest.raises(SessionNotFound):await e.store.create_context_operation('dialogue-a','kkqq','late-context',old.revision)
    with pytest.raises(StaleSessionWrite):await e.store.save_claim(old,'{"late":true}')


async def test_active_run_is_not_deleted(environment):
    e=environment;_,_,body=await e.seed('kkqq','active')
    await e.journal.reserve({**body,'runId':'still-running','messages':[{'id':'new','role':'user','content':'合成新请求'}]},'kkqq','owner')
    with pytest.raises(JournalConflict):await e.service.delete('kkqq','active')
    assert await e.store.load('active') is not None


async def test_failed_secondary_cleanup_can_be_retried_without_revival(environment,monkeypatch):
    e=environment
    await e.seed('kkqq','retry-delete')
    original=e.evidence.delete_session
    monkeypatch.setattr(e.evidence,'delete_session',AsyncMock(side_effect=OSError('隔离清理故障')))
    with pytest.raises(OSError):await e.service.delete('kkqq','retry-delete')
    assert len(await e.journal.sessions('kkqq'))==1
    with pytest.raises(SessionNotFound):await e.store.claim('retry-delete',buyer_id='kkqq')
    monkeypatch.setattr(e.evidence,'delete_session',original)
    await e.service.delete('kkqq','retry-delete')
    assert await e.journal.sessions('kkqq')==[]


async def test_worker_lease_conflict_is_reported_as_409(environment):
    from contextlib import asynccontextmanager
    from app.infrastructure.queue.redis_stream_queue import SessionLeaseTimeout
    e=environment
    await e.seed('kkqq','worker-busy')
    @asynccontextmanager
    async def busy_lease(*args,**kwargs):
        assert kwargs['wait_timeout']==0
        raise SessionLeaseTimeout('对话仍在执行')
        yield
    e.service.orchestrator._session_lease_factory=busy_lease
    headers=await login(e.client,'kkqq')
    response=await e.client.delete('/commerce/ag-ui/sessions/worker-busy',params={'buyer_id':'kkqq'},headers=headers)
    assert response.status_code==409
    assert await e.store.load('worker-busy') is not None
