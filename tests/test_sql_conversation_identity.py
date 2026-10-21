"""当前对话原始主键经journal隔离，标点不能造成串读。"""
import pytest
from app.infrastructure.ag_ui_journal import AGUIJournal,JournalForbidden

async def test_distinct_raw_session_ids_do_not_mix_audit_or_owner(tmp_path):
    journal=AGUIJournal(tmp_path/'runs.db')
    for sid,buyer,text in [('cart:1','one','合成记录一'),('cart1','two','合成记录二')]:
        await journal.reserve({'threadId':sid,'runId':'run-'+sid,'messages':[{'id':'u','role':'user','content':text}]},buyer,'worker')
        await journal.append('run-'+sid,'worker',[{'type':'RUN_FINISHED','threadId':sid,'runId':'run-'+sid}])
    assert (await journal.session('cart:1','one'))['messages'][0]['content']=='合成记录一'
    assert (await journal.session('cart1','two'))['messages'][0]['content']=='合成记录二'
    with pytest.raises(JournalForbidden):await journal.session('cart:1','two')
