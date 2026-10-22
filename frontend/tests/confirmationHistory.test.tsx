import {it,expect} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {CommerceClient} from './clientFixture';
import {mergeConfirmations,readConfirmationDecisions} from '../src/lib/confirmations';
import {readOrderOutcome} from '../src/lib/orderOutcome';
import OrderOutcomeCard from '../src/components/OrderOutcomeCard';
import type {TradeConfirmation} from '../src/types';
const card:TradeConfirmation={confirmation_id:'old-cancel',operation_id:'old-op',action:'cancel',buyer_id:'b1',session_id:'s',
 created_at:'2026-10-21T00:00:00Z',expires_at:'2099-01-01T00:00:00Z',expired:false,status:'pending',result:null,snapshot_hash:'a'.repeat(64),
 payload:{order_id:'order',reason:'合成取消',items:[{product_id:'p',sku_id:'sku',title:'合成商品',quantity:1,unit_price_minor:10000,currency:'CNY'}],
 shipping_address:{recipient_name:'合成买家',country:'CN',state:'',city:'杭州',address_line:'合成地址',postal_code:'',phone:''},
 total_amount_minor:12500,currency:'CNY',subtotal_minor:10000,freight_minor:2500,tariff_minor:0,amount_scope:'landed',order_kind:'purchase_intent'}};
const outcome=readOrderOutcome({order:{order_id:'order',status:'CONFIRMED',currency:'CNY',total_amount_minor:12500,total_amount_major:125,
 created_at:card.created_at,shipping_address:'合成地址',cancel_reason:null,lines:[{product_id:'p',sku_id:'sku',title:'合成商品',quantity:1,unit_price_major:100}]},
 observed_at:card.created_at,operation:{kind:'prepare_cancel',status:'awaiting_confirmation',rejection:null},status:'completed',guidance:'合成快照',
 confirmation_required:true,confirmation:card})!;
const approved:TradeConfirmation={...card,status:'approved',result:{order_id:'order',status:'CANCELLED',total_amount_minor:12500,total_amount_major:125,currency:'CNY',cancel_reason:'合成取消'}};
const recent=Array.from({length:20},(_,i)=>({...card,confirmation_id:`new-${String(i).padStart(2,'0')}`,operation_id:`new-op-${i}`,status:'rejected' as const,created_at:`2026-10-22T00:00:${String(i).padStart(2,'0')}Z`})).reverse();
const presentation={id:'p',runId:'r',kind:'order',status:'completed',payload:outcome};
const json=(data:unknown)=>new Response(JSON.stringify(data),{headers:{'content-type':'application/json'}});
const html=(client:CommerceClient)=>renderToStaticMarkup(<OrderOutcomeCard outcome={outcome} decision={client.getSnapshot().confirmationDecisions.find(d=>d.confirmation_id===card.confirmation_id)}/>);
function history(decision:TradeConfirmation){return {presentations:[presentation],confirmations:recent,confirmationDecisions:readConfirmationDecisions([decision]),
 messages:[{id:'r:final:answer',role:'assistant',content:'合成历史'}],run:{runId:'r',threadId:'s',status:'completed',messages:[],state:{confirmations:recent}}};}

it('历史完整任务按批量决议恢复，原快照保留且不逐订单补读',async()=>{
 const paths:string[]=[];const client=new CommerceClient({buyerId:'b1',url:'/commerce/ag-ui/run',fetch:async(url)=>{
  paths.push(String(url));return String(url).includes('/sessions?')?json({sessions:[{id:'s',title:'合成',updatedAt:1}]}):json(history(approved));
 }});
 await client.initialize();expect(client.getSnapshot().confirmations).toHaveLength(20);
 expect(client.getSnapshot().confirmations.some(c=>c.confirmation_id===card.confirmation_id)).toBe(false);
 expect(html(client)).toContain('取消已确认执行');expect(html(client)).toContain('查看时：已确认');
 expect(outcome.confirmation!.status).toBe('pending');expect(paths.some(p=>p.includes('/orders/')||p.includes('/confirmations/'))).toBe(false);
});

it('重开历史使用当前确认列表，冻结的旧run pending不能恢复确认按钮',async()=>{
 const data={...history(approved),confirmations:[approved],run:{...history(approved).run,state:{confirmations:[card]}}};
 const client=new CommerceClient({buyerId:'b1',url:'/commerce/ag-ui/run',fetch:async(url)=>String(url).includes('/sessions?')
  ?json({sessions:[{id:'s',title:'合成',updatedAt:1}]}):json(data)});
 await client.initialize();expect(client.getSnapshot().confirmations[0].status).toBe('approved');
 expect(html(client)).toContain('取消已确认执行');expect(data.run.state.confirmations[0].status).toBe('pending');
});

it('历史读取期间用户已批准时，迟到历史响应不能恢复pending按钮或脚注',async()=>{
 let finish!:(r:Response)=>void,started!:(v:void)=>void;
 const entered=new Promise<void>(resolve=>{started=resolve;});
 const data={...history(card),confirmations:[card]};
 const client=new CommerceClient({buyerId:'b1',url:'/commerce/ag-ui/run',fetch:async(url,init)=>{
  if(init?.method==='POST')return json({confirmation:approved});
  if(String(url).includes('/sessions?'))return json({sessions:[{id:'s',title:'合成',updatedAt:1}]});
  started();return new Promise<Response>(resolve=>{finish=resolve;});
 }});
 const loading=client.initialize();await entered;
 await client.resolveConfirmation(card,true);finish(json(data));await loading;
 expect(client.getSnapshot().confirmations[0].status).toBe('approved');expect(html(client)).toContain('取消已确认执行');
});

it('刷新在同一次请求带上引用ID，获取列表之外的决议',async()=>{
 const paths:string[]=[];const client=new CommerceClient({buyerId:'b1',url:'/commerce/ag-ui/run',fetch:async(url)=>{
  paths.push(String(url));if(String(url).includes('/confirmations?'))return json({confirmations:recent,confirmationDecisions:readConfirmationDecisions([approved])});
  return String(url).includes('/sessions?')?json({sessions:[{id:'s',title:'合成',updatedAt:1}]}):json(history(card));
 }});
 await client.initialize();expect(html(client)).toContain('取消待确认');await client.refreshConfirmations();
 expect(paths.filter(p=>p.includes('/confirmations?'))).toHaveLength(1);
 expect(paths.at(-1)).toContain('confirmation_ids=old-cancel');expect(html(client)).toContain('取消已确认执行');
});

it('迟到pending刷新不能覆盖已批准决议，其他身份不能替换关联状态',async()=>{
 let finish!:(r:Response)=>void;let refresh=false;
 const client=new CommerceClient({buyerId:'b1',url:'/commerce/ag-ui/run',fetch:async(url,init)=>{
  if(init?.method==='POST')return json({confirmation:approved});
  if(refresh)return new Promise<Response>(resolve=>{finish=resolve;});
  return String(url).includes('/sessions?')?json({sessions:[{id:'s',title:'合成',updatedAt:1}]}):json(history(card));
 }});
 await client.initialize();refresh=true;const pending=client.refreshConfirmations();
 await client.resolveConfirmation(card,true);finish(json({confirmations:recent,confirmationDecisions:readConfirmationDecisions([card])}));await pending;
 expect(html(client)).toContain('取消已确认执行');
});

it('缺少权威决议时显示未核实，不借用当时pending推断尚未执行',()=>{
 const markup=renderToStaticMarkup(<OrderOutcomeCard outcome={outcome}/>);
 expect(markup).toContain('取消决议尚未核实');expect(markup).not.toContain('尚未执行');
});

it('最近记录严格按时间与主键保留，数量与具体ID都正确',()=>{
 const result=mergeConfirmations(recent,[{...recent[0],confirmation_id:'new-20',created_at:'2026-10-22T00:00:20Z'}]);
 expect(result).toHaveLength(20);expect(result[0].confirmation_id).toBe('new-20');
 expect(result.some(c=>c.confirmation_id==='new-19')).toBe(true);expect(result.some(c=>c.confirmation_id==='new-00')).toBe(false);
 const tied=mergeConfirmations([],[{...recent[0],confirmation_id:'a'},{...recent[0],confirmation_id:'z'}]);
 expect(tied.map(c=>c.confirmation_id)).toEqual(['z','a']);
 const precise=mergeConfirmations([],[{...recent[0],confirmation_id:'a',created_at:'2026-10-22T00:00:00.000002Z'},
  {...recent[0],confirmation_id:'z',created_at:'2026-10-22T00:00:00.000001Z'}]);
 expect(precise.map(c=>c.confirmation_id)).toEqual(['a','z']);
});

it('其他买家决议不进入历史卡，读取缺失保持未核实',async()=>{
 const data={...history(card),confirmationDecisions:readConfirmationDecisions([{...approved,buyer_id:'other'}])};
 const client=new CommerceClient({buyerId:'b1',url:'/commerce/ag-ui/run',fetch:async(url)=>String(url).includes('/sessions?')
  ?json({sessions:[{id:'s',title:'合成',updatedAt:1}]}):json(data)});
 await client.initialize();expect(html(client)).toContain('取消决议尚未核实');expect(html(client)).not.toContain('取消已确认执行');
});
