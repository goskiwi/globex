"""只处理已有历史表的显式删除，不参与执行、保存或恢复当前对话。"""
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.infrastructure.persistence.sql.tables import ConversationSessionRow, ConversationMessageRow, ConversationEventRow
from app.domain.session.ports.session_store import SessionOwnerMismatch


async def delete_historical_conversation(engine, session_id: str, buyer_id: str) -> None:
    async with async_sessionmaker(engine)() as db:
        row = await db.get(ConversationSessionRow, session_id)
        if row is not None and row.buyer_id != buyer_id:
            raise SessionOwnerMismatch('无权删除其他买家的对话')
        for model in (ConversationEventRow, ConversationMessageRow, ConversationSessionRow):
            await db.execute(delete(model).where(model.session_id == session_id))
        await db.commit()
