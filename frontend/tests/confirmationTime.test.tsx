// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import ConfirmationCards from '../src/components/ConfirmationCards';
import OrderOutcomeCard from '../src/components/OrderOutcomeCard';
import type {OrderOutcome,TradeConfirmation} from '../src/types';

it('同一份pending回执自然到期时，两类实际组件同步更新；批准终态不随时间倒退',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  vi.useFakeTimers();vi.setSystemTime(new Date('2030-01-01T00:00:00Z'));
  const intervals=vi.spyOn(globalThis,'setInterval'), cleared=vi.spyOn(globalThis,'clearInterval');
  const c:TradeConfirmation={confirmation_id:'c',operation_id:'op',buyer_id:'b',session_id:'s',action:'cancel',
    status:'pending',expired:false,created_at:'2030-01-01T00:00:00Z',expires_at:'2030-01-01T00:00:02Z',
    snapshot_hash:'a'.repeat(64),result:null,payload:{order_id:'o',reason:'合成取消',items:[],
      shipping_address:{recipient_name:'合成买家',country:'CN',state:'',city:'杭州',address_line:'合成地址',postal_code:'',phone:''},
      currency:'CNY',subtotal_minor:100,freight_minor:0,tariff_minor:0,total_amount_minor:100,amount_scope:'landed',order_kind:'purchase_intent'}};
  const outcome:OrderOutcome={order:{order_id:'o',status:'CONFIRMED',currency:'CNY',total_amount_minor:100,total_amount_major:1,
    lines:[],created_at:c.created_at,shipping_address:'合成地址',cancel_reason:null},observed_at:c.created_at,
    status:'completed',guidance:'',operation:{kind:'prepare_cancel',status:'awaiting_confirmation',rejection:null},confirmation:c,confirmation_required:true};
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  const render=(current:TradeConfirmation)=> <><ConfirmationCards confirmations={[current]} busy={false} error={null}
      onResolve={()=>{}} onCancelOrder={()=>{}} onRefresh={()=>{}}/><OrderOutcomeCard outcome={outcome} decision={current}/></>;
  try{
    await act(async()=>root.render(render(c)));
    expect(host.textContent).toContain('等待你确认');expect(host.textContent).toContain('取消待确认，尚未执行');
    expect(intervals).toHaveBeenCalledTimes(1);
    await act(async()=>vi.advanceTimersByTime(3000));
    expect(host.textContent).toContain('确认已过期');expect(host.textContent).toContain('取消确认已过期，尚未执行');
    expect(host.textContent).not.toContain('取消待确认，尚未执行');expect(c.expired).toBe(false);
    const approved={...c,status:'approved' as const,result:{order_id:'o',status:'CANCELLED' as const,total_amount_minor:100,total_amount_major:1,currency:'CNY',cancel_reason:'合成取消'}};
    await act(async()=>root.render(render(approved)));
    expect(host.textContent).toContain('取消已确认执行');expect(cleared).toHaveBeenCalledWith(intervals.mock.results[0].value);
    await act(async()=>vi.advanceTimersByTime(60000));
    expect(host.textContent).toContain('取消已确认执行');expect(host.textContent).not.toContain('取消确认已过期');
  }finally{
    await act(async()=>root.unmount());host.remove();vi.restoreAllMocks();vi.useRealTimers();
  }
});
