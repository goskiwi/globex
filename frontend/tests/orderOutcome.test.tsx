import {renderToStaticMarkup} from 'react-dom/server';
import {expect,it} from 'vitest';
import OrderOutcomeCard from '../src/components/OrderOutcomeCard';
import {readOrderOutcome} from '../src/lib/orderOutcome';
import {readPresentations} from '../src/lib/commerceClient';
const base={order:{order_id:'synthetic-order',status:'CONFIRMED',currency:'CNY',total_amount_minor:15400,total_amount_major:154,
 created_at:'2026-10-21T00:00:00Z',shipping_address:'合成地址',cancel_reason:null,
 lines:[{product_id:'P1003',sku_id:'P1003-S1',title:'合成背包',unit_price_major:129,quantity:1}]},
 observed_at:'2026-10-22T00:00:00Z',status:'completed',guidance:'订单状态来自程序。',
 operation:{kind:'inspect',status:'completed',rejection:null},confirmation_required:false,confirmation:null};
it('纯查询显示完整订单快照，不生成取消确认或使用模型重写金额',()=>{
 const parsed=readOrderOutcome(base)!;expect(parsed).not.toBeNull();
 const html=renderToStaticMarkup(<OrderOutcomeCard outcome={parsed}/>);
 expect(html).toContain('查看时：已确认');expect(html).toContain('154.00');expect(html).toContain('合成背包');
 expect(html).toContain('下单时间');expect(html).toContain('仅查看订单');expect(html).not.toContain('取消待确认');
});
it('取消被拒绝仍展示已授权订单状态及拒绝原因',()=>{
 const parsed=readOrderOutcome({...base,status:'partial',order:{...base.order,status:'CANCELLED'},
  operation:{kind:'prepare_cancel',status:'rejected',rejection:{code:'ORDER_CHANGED',message:'只有未取消的订单可以准备取消'}}})!;
 expect(parsed).not.toBeNull();const html=renderToStaticMarkup(<OrderOutcomeCard outcome={parsed}/>);
 expect(html).toContain('查看时：已取消');expect(html).toContain('取消未准备');expect(html).toContain('只有未取消');
});
it('结构化订单展示在会话恢复时保留，缺事实或伪造确认不会接纳',()=>{
 expect(readPresentations([{id:'p',runId:'r',kind:'order',status:'completed',payload:base}])).toHaveLength(1);
 expect(readOrderOutcome({...base,order:{...base.order,created_at:undefined}})).toBeNull();
 expect(readOrderOutcome({...base,operation:{kind:'prepare_cancel',status:'awaiting_confirmation',rejection:null},confirmation_required:true})).toBeNull();
});
it('取消准备后的快照保留查询时状态，操作状态跟随最新确认结果',()=>{
 const confirmation={confirmation_id:'c',operation_id:'op',action:'cancel',buyer_id:'b',session_id:'s',created_at:'2026-10-22T00:00:00Z',snapshot_hash:'a'.repeat(64),
  expires_at:'2099-01-01T00:00:00Z',status:'pending',result:null,expired:false,
  payload:{order_id:base.order.order_id,reason:'合成取消',items:[{product_id:'P1003',sku_id:'P1003-S1',title:'合成背包',quantity:1,unit_price_minor:12900,currency:'CNY'}],
   shipping_address:{recipient_name:'合成买家',country:'CN',state:'',city:'杭州',address_line:'合成地址',postal_code:'',phone:''},
   currency:'CNY',subtotal_minor:12900,freight_minor:2500,tariff_minor:0,total_amount_minor:15400,amount_scope:'landed',order_kind:'purchase_intent'}};
 const outcome=readOrderOutcome({...base,operation:{kind:'prepare_cancel',status:'awaiting_confirmation',rejection:null},confirmation_required:true,confirmation})!;
 expect(outcome).not.toBeNull();
 const approved={...outcome.confirmation!,status:'approved' as const,result:{order_id:base.order.order_id,status:'CANCELLED' as const,total_amount_minor:15400,total_amount_major:154,currency:'CNY',cancel_reason:'合成取消'}};
 const html=renderToStaticMarkup(<OrderOutcomeCard outcome={outcome} decision={approved}/>);
 expect(html).toContain('查看时：已确认');expect(html).toContain('取消已确认执行');expect(html).not.toContain('取消待确认，尚未执行');
 const expired=renderToStaticMarkup(<OrderOutcomeCard outcome={outcome} decision={{...outcome.confirmation!,expired:true}}/>);
 expect(expired).toContain('取消确认已过期');
});
