import type {OrderOutcome} from '../types';
import {readConfirmations} from './confirmations';
const record=(v:unknown):v is Record<string,any>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const amount=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
export function readOrderOutcome(value:unknown):OrderOutcome|null {
 if(!record(value)||!record(value.order)||!record(value.operation)||typeof value.guidance!=='string'
  ||!['completed','partial'].includes(value.status)||typeof value.observed_at!=='string'||!Number.isFinite(Date.parse(value.observed_at)))return null;
 const o=value.order,a=value.operation;
 if(typeof o.order_id!=='string'||!['DRAFT','CONFIRMED','CANCELLED'].includes(o.status)||typeof o.currency!=='string'
  ||!Number.isSafeInteger(o.total_amount_minor)||o.total_amount_minor<0||!amount(o.total_amount_major)
  ||typeof o.shipping_address!=='string'||typeof o.created_at!=='string'||!Number.isFinite(Date.parse(o.created_at))
  ||!(o.cancel_reason===null||typeof o.cancel_reason==='string')||!Array.isArray(o.lines)||!o.lines.length
  ||o.lines.some((l:unknown)=>!record(l)||typeof l.product_id!=='string'||typeof l.sku_id!=='string'||typeof l.title!=='string'
   ||!amount(l.unit_price_major)||!Number.isSafeInteger(l.quantity)||l.quantity<=0))return null;
 if(!['inspect','prepare_cancel'].includes(a.kind)||!['completed','awaiting_confirmation','rejected'].includes(a.status))return null;
 if(a.status==='rejected'){
  if(!record(a.rejection)||typeof a.rejection.code!=='string'||typeof a.rejection.message!=='string'||value.status!=='partial')return null;
 }else if(a.rejection!==null||value.status!=='completed')return null;
 const cards=value.confirmation===null?[]:readConfirmations([value.confirmation]);
 if(a.status==='awaiting_confirmation'){
  if(a.kind!=='prepare_cancel'||value.confirmation_required!==true||cards.length!==1||cards[0].action!=='cancel'
   ||cards[0].payload.order_id!==o.order_id||cards[0].status!=='pending')return null;
 }else if(value.confirmation_required!==false||value.confirmation!==null)return null;
 if(a.kind==='inspect'&&a.status!=='completed')return null;
 return {...value,confirmation:cards[0]??null} as OrderOutcome;
}
