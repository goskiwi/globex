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
