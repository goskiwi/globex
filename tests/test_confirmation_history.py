"""完整历史成果的决议批量恢复不依赖最近列表，不预读订单。"""
from datetime import timedelta
from types import SimpleNamespace
import pytest
from sqlalchemy import event
from app.application.usecases.order_usecases import OrderItemInput,HandleOrderUseCase
from app.application.usecases.confirmation_views import recent_confirmations
from app.application.runtime.runs import RunService
from app.infrastructure.ag_ui_journal import AGUIJournal
from app.infrastructure.ag_ui_journal import JournalForbidden
from tests.trade_test_helpers import confirmation_env,test_address as address
from tests.test_order_task import create
from tests.test_ag_ui_journal import body


@pytest.mark.parametrize('decision',['approved','rejected','expired'])
async def test_session_restores_referenced_decision_after_history_limit(confirmation_env,tmp_path,decision):
    e=confirmation_env;order=await create(e)
    outcome=await HandleOrderUseCase(e.service).execute(order['order_id'],buyer_id='buyer',session_id='session',cancellation={'reason':'合成取消'})
    card=outcome['confirmation']
    if decision!='expired':await e.service.resolve(card['confirmation_id'],'buyer','session',card['snapshot_hash'],decision=='approved')
    else:e.now[0]+=timedelta(seconds=600)
    for _ in range(21):
        e.now[0]+=timedelta(seconds=1)
        p=await e.service.prepare_order('buyer','session',[OrderItemInput('P1003','P1003-S1',1)],address(),'CNY')
        c=p['confirmation'];await e.service.resolve(c['confirmation_id'],'buyer','session',c['snapshot_hash'],False)
    assert card['confirmation_id'] not in {c['confirmation_id'] for c in (await e.service.restore('buyer','session'))['confirmations']}
    journal=AGUIJournal(tmp_path/'history.db');request=body('history');request['threadId']='session'
    saved,_=await journal.reserve(request,'buyer','owner')
    await journal.append('history','owner',[{'type':'STATE_SNAPSHOT','snapshot':{'presentations':[
        {'id':'p','runId':'history','kind':'order','status':'completed','payload':outcome}], 'confirmations':[card]}},
        {'type':'RUN_FINISHED','runId':'history','threadId':'session'}])
    reads=[]
    def measured(_c,_cu,statement,_pa,_co,_many):
        if statement.lstrip().startswith('SELECT') and 'trade_confirmations' in statement:reads.append(statement)
        assert 'FROM orders' not in statement
    event.listen(e.engine.sync_engine,'before_cursor_execute',measured)
    try:restored=await RunService(journal,SimpleNamespace(),e.service).session('session','buyer')
    finally:event.remove(e.engine.sync_engine,'before_cursor_execute',measured)
    values=restored['confirmationDecisions'];assert len(values)==1 and len(reads)==3 # 列表2查询、全部引用1查询，均在同一快照
    assert values[0]['confirmation_id']==card['confirmation_id']
    assert values[0]['status']==('pending' if decision=='expired' else decision)
    assert values[0]['expired'] is (decision=='expired')
    assert set(values[0])=={'confirmation_id','buyer_id','session_id','action','status','expires_at','expired'}
    assert restored['presentations'][0]['payload']==outcome, '历史快照不因新决议被改写'
    assert restored['run']['state']['confirmations'][0]['status']=='pending'
    assert len(restored['confirmations'])==20 and all(x['confirmation_id']!=card['confirmation_id'] for x in restored['confirmations'])
    assert not (await e.service.restore('other','session',[card['confirmation_id']]))['confirmationDecisions']
    assert not (await e.service.restore('buyer','other',[card['confirmation_id']]))['confirmationDecisions']
    with pytest.raises(JournalForbidden):await RunService(journal,None,e.service).session('session','other')


def test_recent_selection_is_chronological_monotonic_and_preserves_receipts():
    from datetime import datetime,timezone
    def card(i,status='rejected'):
        return {'confirmation_id':f'c{i:02}','created_at':'2026-10-22T00:00:00+00:00','expires_at':'2099-01-01T00:00:00+00:00',
                'status':status,'expired':False,'result':None}
    original=[card(i) for i in range(20,0,-1)]
    merged=recent_confirmations(original,[card(21)],now=datetime.now(timezone.utc))
    assert [c['confirmation_id'] for c in merged]==[f'c{i:02}' for i in range(21,1,-1)]
    approved={**card(0,'approved'),'result':{'order_id':'order','status':'CONFIRMED'}}
    result=recent_confirmations([approved,*original],[card(0,'pending')])
    assert not any(c['confirmation_id']=='c00' for c in result), '批准不能被迟到pending恢复成有效操作'
    assert approved['result']['status']=='CONFIRMED'
    many=recent_confirmations([card(i,'pending') for i in range(25)],original)
    assert len([c for c in many if c['status']=='pending'])==5 # 同ID的终态优先，另5个有效操作保留
