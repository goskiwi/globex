"""失败事实在实时流、单run、跨轮历史和新进程恢复中一致；不补造成功消息。"""
import json
import pytest
from ag_ui.core import RunAgentInput
from app.infrastructure.ag_ui_journal import AGUIJournal,JournalForbidden
from app.application.agents.ag_ui_adapter import AGUIRunAdapter
from tests.test_ag_ui_journal import body
from scripts.run_stream import read_stream


@pytest.mark.parametrize('origin,code',[
    ('adapter','PERSISTENCE_FAILED'),('adapter','CONTEXT_CAPACITY_EXCEEDED'),
    ('adapter','SESSION_VERSION_CHANGED'),('adapter','CANCELLED'),
    ('finish','output_limit'),('shutdown','SERVER_RESTART'),('expire','SERVER_RESTART'),('queued','CANCELLED')])
async def test_failure_survives_replay_later_turn_and_process_restart(tmp_path,origin,code):
    path=tmp_path/'journal.db';journal=AGUIJournal(path)
    saved,_=await journal.reserve(body('failed'),'b1','owner',queued=origin=='queued')
    if origin=='queued':await journal.request_stop('failed','b1')
    else:
        emitted=[];adapter=AGUIRunAdapter(RunAgentInput.model_validate(saved['input']),emitted.append)
        adapter.start()
        if origin=='adapter':adapter.fail('合成故障：操作可能已生效，请先核查，不直接重放。',cancelled=code=='CANCELLED',code=code)
        elif origin=='finish':adapter.finish('合成失败，未取得完整结果。','failed',code)
        await journal.append('failed','owner',[e.model_dump(mode='json',by_alias=True,exclude_none=True) for e in emitted])
        if origin=='shutdown':await journal.end_owned('failed','owner')
        elif origin=='expire':
            async with journal._db(True) as db:await db.execute("UPDATE agui_runs SET lease_until=0 WHERE run_id='failed'")
            await journal.recover_expired()
    failed=await journal.run('failed','b1');before,_,_=await journal.events('failed','b1')
    frames=[r['event'] for r in before]
    stream=read_stream(''.join('data: '+json.dumps(f)+'\n\n' for f in frames))
    error=failed['state']['process']['error']
    assert stream.error==error['message'] and error['code']==code
    assert frames[-2]['snapshot']['process']['error']==error
    assert failed['result']['text']==error['message']
    assert failed['result']['status']==('partial' if code=='CANCELLED' else 'failed')
    assert not any(m['role']=='assistant' for m in failed['messages']) if origin!='finish' else True
    saved,_=await journal.reserve(body('next'),'b1','next-owner')
    emitted=[];adapter=AGUIRunAdapter(RunAgentInput.model_validate(saved['input']),emitted.append)
    adapter.start();adapter.finish('下一轮合成答复','completed',None)
    await journal.append('next','next-owner',[e.model_dump(mode='json',by_alias=True,exclude_none=True) for e in emitted])
    restored=await AGUIJournal(path).session('s1','b1')
    assert restored['run']['status']=='completed'
    assert restored['processes'][0]['error']==error
    assert 'error' not in restored['processes'][1]
    assert (await journal.events('failed','b1'))[0]==before
    with pytest.raises(JournalForbidden):await journal.session('s1','another-buyer')


async def test_many_followups_do_not_remove_old_failure_reason(tmp_path):
    journal=AGUIJournal(tmp_path/'journal.db')
    for i in range(55):
        saved,_=await journal.reserve(body(str(i)),'b1','owner')
        events=[];adapter=AGUIRunAdapter(RunAgentInput.model_validate(saved['input']),events.append)
        adapter.start()
        if i==0:adapter.fail('首轮合成保存失败，请核查后再操作。',code='PERSISTENCE_FAILED')
        else:adapter.finish(f'第{i}轮成功','completed',None)
        await journal.append(str(i),'owner',[e.model_dump(mode='json',by_alias=True,exclude_none=True) for e in events])
    restored=await AGUIJournal(journal.path).session('s1','b1')
    assert len(restored['processes'])==55 and len(restored['events'])==80
    assert restored['processes'][0]['error']['code']=='PERSISTENCE_FAILED'
    assert restored['processes'][0]['error']['message']=='首轮合成保存失败，请核查后再操作。'
