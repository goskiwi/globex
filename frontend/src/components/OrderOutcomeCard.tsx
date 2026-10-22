import type {OrderOutcome,TradeConfirmation} from '../types';
import {money} from './ProductCards';
import './myOrders.css';
const statuses={DRAFT:'草稿',CONFIRMED:'已确认',CANCELLED:'已取消'};
export default function OrderOutcomeCard({outcome,confirmation}:{outcome:OrderOutcome;confirmation?:TradeConfirmation}){
 const o=outcome.order,a=outcome.operation,c=confirmation??outcome.confirmation;
 const action=a.status==='awaiting_confirmation'
  ?c?.status==='approved'?'取消已确认执行':c?.status==='rejected'?'取消已拒绝':c?.expired?'取消确认已过期，尚未执行':'取消待确认，尚未执行'
  :a.status==='rejected'?'取消未准备':'仅查看订单';
 return <section className="order-list" aria-label="订单查看结果"><article className="order-card">
  <header><div><small>查看快照 · {new Date(outcome.observed_at).toLocaleString('zh-CN')}</small><h2>{o.order_id}</h2></div>
   <span className={`order-status ${o.status}`}>查看时：{statuses[o.status]}</span></header>
  <div className="order-lines">{o.lines.map(l=><div key={l.sku_id}><div><strong>{l.title}</strong><small>{l.sku_id} · 数量 {l.quantity}</small></div>
   <span>{money(l.unit_price_major,o.currency)} / 件</span></div>)}</div>
  <footer><span>到手总额 <strong>{money(o.total_amount_minor/100,o.currency)}</strong></span>
   <span>{action}</span></footer>
  <div className="order-detail"><p>下单时间：{new Date(o.created_at).toLocaleString('zh-CN')}</p><p>收货信息：{o.shipping_address}</p>
   {a.rejection&&<p role="alert">{a.rejection.message}</p>}{o.cancel_reason&&<p>取消原因：{o.cancel_reason}</p>}</div>
 </article></section>;
}
