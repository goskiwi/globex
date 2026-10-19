"""会话主键原样贯穿快照、对话和买家归属，标点不能造成串读。"""
from app.infrastructure.persistence.sql.repositories import SqlConversationStore
from app.domain.session.ports.conversation_store import ConversationTurn
from tests.sql_fixtures import session_store

async def test_distinct_raw_session_ids_do_not_mix_audit_or_owner(tmp_path):
    sessions=await session_store(tmp_path)
    audit=SqlConversationStore(sessions._engine)
    for sid,buyer,text in [('cart:1','one','合成记录一'),('cart1','two','合成记录二')]:
        await sessions.assert_owner(sid,buyer,create=True)
        await audit.touch_session(sid,buyer,'zh-CN','CNY')
        await audit.append_turn(ConversationTurn(sid,buyer,'buyer',text))
    assert [t.content for t in await audit.list_turns('cart:1')]==['合成记录一']
    assert [t.buyer_id for t in await audit.list_turns('cart1')]==['two']
    assert (await audit.find_session('cart:1'))['buyer_id']=='one'
    assert (await audit.find_session('cart1'))['buyer_id']=='two'
