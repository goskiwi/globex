"""隔离 SQL 存储工厂；返回生产仓库类型，连接由测试 fixture 统一释放。"""
from app.infrastructure.persistence.sql.repositories import create_engine,bootstrap_schema,SqlPreferenceStore,SqlSessionStore
engines=[]
async def make_store(path,cls):
    path.mkdir(parents=True,exist_ok=True)
    engine=create_engine('sqlite+aiosqlite:///'+str(path/'stores.db'))
    engines.append(engine)
    await bootstrap_schema(engine)
    return cls(engine)
async def preference_store(path):return await make_store(path,SqlPreferenceStore)
async def session_store(path):return await make_store(path,SqlSessionStore)

async def read_session_state(store, session_id):
    from app.infrastructure.persistence.sql.tables import AgentSessionStateRow
    await store.initialize()
    async with store._sessions() as db:
        row = await db.get(AgentSessionStateRow, session_id)
        return row.state_json if row else None

async def save_session_state(store, session_id, state_json):
    claim = await store.claim(session_id, buyer_id="buyer")
    return await store.save_claim(claim, state_json)

async def seed_historical_state(store, session_id, state_json, owner=None):
    from app.infrastructure.persistence.sql.tables import AgentSessionStateRow, ConversationSessionRow
    async with store._transaction() as db:
        db.add(AgentSessionStateRow(session_id=session_id, state_json=state_json))
        if owner:
            db.add(ConversationSessionRow(session_id=session_id, buyer_id=owner))

async def seed_historical_conversation(engine, session_id, buyer_id, content='合成历史正文'):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.infrastructure.persistence.sql.tables import ConversationSessionRow, ConversationMessageRow, ConversationEventRow
    async with async_sessionmaker(engine)() as db:
        db.add(ConversationSessionRow(session_id=session_id,buyer_id=buyer_id))
        db.add(ConversationMessageRow(session_id=session_id,turn_index=1,buyer_id=buyer_id,role='buyer',content=content))
        db.add(ConversationEventRow(session_id=session_id,type='historical',payload={'synthetic':True},occurred_at='2026-01-01T00:00:00Z'))
        await db.commit()
