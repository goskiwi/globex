"""当前执行只保存journal；历史数据保留和显式删除仍受归属保护。"""
import json
import pytest
from sqlalchemy import select,func,event
from sqlalchemy.ext.asyncio import async_sessionmaker
from ag_ui.core import RunAgentInput
from langchain_core.messages import AIMessage
from app.infrastructure.persistence.sql.historical_conversations import delete_historical_conversation
from app.infrastructure.persistence.sql.tables import ConversationSessionRow,ConversationMessageRow,ConversationEventRow
from app.domain.session.ports.session_store import SessionOwnerMismatch
from tests.sql_fixtures import seed_historical_conversation,session_store
from tests.test_langgraph_runtime import _container
from tests.test_agent_handoff import ScriptedModel

async def test_run_preserves_history_but_never_dual_writes_legacy_tables(tmp_path,monkeypatch):
    c=await _container(tmp_path,monkeypatch,ScriptedModel(responses=[AIMessage(content='合成回复')]))
    inserts=[]
    try:
        await seed_historical_conversation(c.db_engine,'historical','old-buyer')
        def observe(_conn,_cursor,statement,_params,_context,_many):
            if statement.lower().startswith('insert into conversation_'):inserts.append(statement)
        event.listen(c.db_engine.sync_engine,'before_cursor_execute',observe)
        await c.runs.startup()
        body=RunAgentInput.model_validate({'threadId':'current','runId':'current-run','state':{},'tools':[],'context':[],
            'messages':[{'id':'u','role':'user','content':'合成请求'}],'forwardedProps':{'buyerId':'buyer'}})
        await c.runs.start(body,'buyer');run=await c.runs.wait('current-run','buyer',10)
        assert run['status']=='completed' and not inserts
        assert [m['content'] for m in (await c.runs.journal.session('current','buyer'))['messages']]==['合成请求','合成回复']
        frames,_,_=await c.runs.journal.events('current-run','buyer')
        assert frames[-1]['event']['type']=='RUN_FINISHED'
        assert any(e['event'].get('name')=='usage.summary' for e in frames)
        async with async_sessionmaker(c.db_engine)() as db:
            for table in [ConversationSessionRow,ConversationMessageRow,ConversationEventRow]:
                assert await db.scalar(select(func.count()).select_from(table))==1
        assert not hasattr(c.orchestrator,'_conversation_store')
    finally:await c.shutdown()

async def test_historical_cleanup_is_owned_and_does_not_touch_other_sessions(tmp_path):
    store=await session_store(tmp_path)
    await seed_historical_conversation(store._engine,'old-one','one')
    await seed_historical_conversation(store._engine,'old-two','two')
    with pytest.raises(SessionOwnerMismatch):
        await delete_historical_conversation(store._engine,'old-one','two')
    async with async_sessionmaker(store._engine)() as db:
        assert await db.scalar(select(func.count()).select_from(ConversationMessageRow))==2
    await delete_historical_conversation(store._engine,'old-one','one')
    async with async_sessionmaker(store._engine)() as db:
        for table in [ConversationSessionRow,ConversationMessageRow,ConversationEventRow]:
            rows=list(await db.scalars(select(table)))
            assert len(rows)==1 and rows[0].session_id=='old-two'
