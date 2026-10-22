"""临时探针：刷新与独立用户批准并行时，列表和引用决议应来自同一视图。"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from tests.trade_test_helpers import confirmation_env
from tests.test_order_task import create

async def test_restore_decision_snapshot(confirmation_env,monkeypatch):
 e=confirmation_env;o=await create(e)
 c=(await e.service.handle_order('buyer','session',o['order_id'],{'reason':'合成并发'}))['confirmation']
 entered,release=asyncio.Event(),asyncio.Event();original=AsyncSession.scalars
 async def paused(db,statement,*args,**kw):
  result=await original(db,statement,*args,**kw)
  if 'trade_confirmations' in str(statement) and not entered.is_set():
   entered.set();await release.wait()
  return result
 monkeypatch.setattr(AsyncSession,'scalars',paused)
 reader=asyncio.create_task(e.service.restore('buyer','session',[c['confirmation_id']]))
 try:
  await asyncio.wait_for(entered.wait(),3)
  await asyncio.wait_for(e.service.resolve(c['confirmation_id'],'buyer','session',c['snapshot_hash'],True),3)
 finally:release.set()
 result=await reader
 full=next(x for x in result['confirmations'] if x['confirmation_id']==c['confirmation_id'])
 decision=result['confirmationDecisions'][0]
 assert full['status']==decision['status']=='pending'
 fresh=await e.service.restore('buyer','session',[c['confirmation_id']])
 assert next(x for x in fresh['confirmations'] if x['confirmation_id']==c['confirmation_id'])['status']==fresh['confirmationDecisions'][0]['status']=='approved'
