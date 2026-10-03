// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import ProductDetail from '../src/components/ProductDetail';
import {groupProductRecords} from '../src/lib/productGroups';
import {readProducts} from '../src/lib/commerceClient';
import type {ProductCard} from '../src/types';
import App from './WorkspaceFixture';

const records:ProductCard[]=['amazon','ebay'].map((source_platform,index)=>({
  product_id:`P302${index}`,canonical_product_id:'CAN-GX-02-2',title:'Roamix 单侧支撑颈枕',brand:'Roamix',category:'旅行装备',
  origin_country:'US',source_platform,price_major:198,currency:'CNY',score:1,highlights:[],description:'可拆洗纯棉外套。',
  default_sku_id:`P302${index}-S1`,ships_to:['CN'],weight_kg:index?0.17:0.15,rating_summary:{average:index?4.3:4.2,review_count:80+index},
  skus:[1,2].map(n=>({sku_id:`P302${index}-S${n}`,spec:n===1?'石墨黑':'雾灰',price_major:index?30.14:198+n-1,
    currency:index?'USD':'CNY',stock:index&&n===2?95:0,display_price_major:index?213.99:198+n-1,display_currency:'CNY',constraint_issues:['over_price_cap']})),
}));

it('详情消费服务端已选规格，不能被最低价默认项覆盖，准备下单沿用同一规格',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  const onPrepare=vi.fn();
  const selected=readProducts([{...records[0],default_sku_id:'P3020-S2',selected_sku_id:'P3020-S2',
    skus:records[0].skus.map(sku=>({...sku,stock:10}))}]);
  try{
    await act(async()=>root.render(<ProductDetail group={groupProductRecords(selected)[0]} busy={false}
      onClose={()=>{}} onPrepare={onPrepare} onCompare={()=>{}} onAsk={()=>{}}/>));
    expect(host.querySelector<HTMLInputElement>('input[name="product-sku"]:checked')!.value).toBe('P3020-S2');
    expect(host.querySelectorAll('input[name="product-sku"]')).toHaveLength(2);
    await act(async()=>[...host.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent==='准备下单')!.click());
    expect(onPrepare).toHaveBeenCalledWith(selected[0],'P3020-S2','CNY');
  }finally{await act(async()=>root.unmount());host.remove();}
});

it('选择真实平台和规格，比较及购买使用对应ID；缺货可看可比较但不能买，超预算不清预算',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  const onPrepare=vi.fn(),onCompare=vi.fn(),onAsk=vi.fn();
  const before=JSON.stringify(records);
  const button=(text:string)=>[...host.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent===text)!;
  try{
    await act(async()=>root.render(<ProductDetail group={groupProductRecords(records)[0]} busy={false} onClose={()=>{}}
      onPrepare={onPrepare} onCompare={onCompare} onAsk={onAsk}/>));
    expect(host.querySelector<HTMLInputElement>('input[name="platform"]:checked')!.value).toBe('P3021');
    expect(host.querySelector<HTMLInputElement>('input[name="product-sku"]:checked')!.value).toBe('P3021-S2');
    expect(host.textContent).toContain('4.3');expect(host.textContent).toContain('0.17 kg');
    expect(host.textContent).toContain('213.99');expect(host.textContent).toContain('30.14');
    expect(button('准备下单').disabled).toBe(false);
    expect(onPrepare).not.toHaveBeenCalled();
    await act(async()=>button('加入比较').click());
    expect(onCompare.mock.calls[0][0]).toMatchObject({product_id:'P3021',default_sku_id:'P3021-S2',price_major:213.99,currency:'CNY'});
    await act(async()=>button('准备下单').click());
    expect(onPrepare).toHaveBeenCalledWith(records[1],'P3021-S2','CNY');
    await act(async()=>host.querySelector<HTMLInputElement>('input[name="platform"][value="P3020"]')!.click());
    expect(host.textContent).toContain('4.2');expect(host.textContent).toContain('0.15 kg');
    expect(button('当前规格缺货').disabled).toBe(true);
    await act(async()=>host.querySelector<HTMLInputElement>('input[name="product-sku"][value="P3020-S2"]')!.click());
    await act(async()=>button('加入比较').click());
    expect(onCompare.mock.calls[1][0]).toMatchObject({product_id:'P3020',default_sku_id:'P3020-S2',price_major:199,currency:'CNY'});
    expect(onPrepare).toHaveBeenCalledTimes(1);expect(onAsk).not.toHaveBeenCalled();
    expect(JSON.stringify(records)).toBe(before);
  }finally{await act(async()=>root.unmount());host.remove();}
});

it('推荐抽屉保留交付规格与有效到手价，改规格后不沿用旧报价和理由',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const product:ProductCard={...records[0],default_sku_id:'P3020-S2',recommendation_reason:'当前交付理由',quantity:1,
    skus:records[0].skus.map(s=>({...s,stock:10})),landed_price:{items:[{product_id:'P3020',sku_id:'P3020-S2',quantity:1,title:'颈枕',
      unit_price_minor:19900,source_unit_price_minor:19900,source_currency:'CNY',currency:'CNY',subtotal_minor:19900,freight_minor:2500,tariff_minor:0,total_amount_minor:22400}],
      ship_to:'CN',currency:'CNY',subtotal_minor:19900,freight_minor:2500,tariff_minor:0,total_amount_minor:22400}};
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);const compare=vi.fn();
  try{
    await act(async()=>root.render(<ProductDetail group={{...groupProductRecords([product])[0],initialSkuId:product.default_sku_id}} busy={false} onClose={()=>{}}
      onPrepare={()=>{}} onCompare={compare} onAsk={()=>{}}/>));
    expect(host.querySelector<HTMLInputElement>('input[name="product-sku"]:checked')!.value).toBe('P3020-S2');
    expect(host.querySelector('.landed-detail-panel')!.textContent).toContain('224.00');
    await act(async()=>host.querySelector<HTMLInputElement>('input[value="P3020-S1"]')!.click());
    expect(host.querySelector('.landed-detail-panel')).toBeNull();
    await act(async()=>[...host.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent==='加入比较')!.click());
    expect(compare.mock.calls[0][0].landed_price).toBeUndefined();
    expect(compare.mock.calls[0][0].recommendation_reason).toBeUndefined();
  }finally{await act(async()=>root.unmount());host.remove();}
});

it('实际App查看到确认闭环传递精确平台SKU与展示币种，准备不创建订单，明确确认后才执行',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  const products=records.map(p=>({...p,currency:p.source_platform==='ebay'?'USD':'CNY'}));
  let prepared:any=null,orderCount=0,session='',run='';const requests:{url:string;body:any}[]=[];
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    const body=init?.body?JSON.parse(String(init.body)):null;requests.push({url:String(url),body});
    if(url.endsWith('/ag-ui/run')){
      session=body.threadId;run=body.runId;
      const events=[{type:'RUN_STARTED',threadId:session,runId:run},
        {type:'MESSAGES_SNAPSHOT',messages:[...body.messages,{id:`${run}:final:answer`,role:'assistant',content:'查看这款颈枕。'}]},
        {type:'STATE_SNAPSHOT',snapshot:{productViews:[{runId:run,guidance:'查看这款颈枕。',hits:products,result_ref:'ctx_view',purpose:'product_view'}]}},
        {type:'RUN_FINISHED',threadId:session,runId:run}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.endsWith('/confirmations/orders')){
      prepared={confirmation_id:'confirm-view',operation_id:'operation-view',buyer_id:'test-buyer',session_id:session,action:'create',
        status:'pending',result:null,snapshot_hash:'test-snapshot',expires_at:new Date(Date.now()+300000).toISOString(),expired:false,
        payload:{items:[{...body.items[0],title:'Roamix 雾灰',unit_price_minor:21399,currency:'CNY'}],shipping_address:body.shipping_address,
          currency:'CNY',subtotal_minor:21399,freight_minor:2500,tariff_minor:0,total_amount_minor:23899,amount_scope:'landed',order_kind:'purchase_intent'}};
      return Response.json({confirmation_required:true,confirmation:prepared});
    }
    if(url.endsWith('/confirmations/confirm-view/resolve')){
      expect(body.approved).toBe(true);orderCount++;
      prepared={...prepared,status:'approved',result:{order_id:'GBX-view',status:'CONFIRMED',total_amount_minor:23899,total_amount_major:238.99,currency:'CNY',cancel_reason:null}};
      return Response.json({confirmation_required:false,confirmation:prepared,order:prepared.result});
    }
    return Response.json({sessions:[],confirmations:prepared?[prepared]:[],skills:[],products:[],form:null,revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  async function input(selector:string,value:string){await act(async()=>{
    const el=host.querySelector<HTMLInputElement|HTMLTextAreaElement>(selector)!;
    const prototype=el instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype,'value')!.set!.call(el,value);el.dispatchEvent(new Event('input',{bubbles:true}));
  });}
  const button=(text:string)=>[...host.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent===text)!;
  try{
    await act(async()=>root.render(<App/>));await input('#query','查看颈枕');
    await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="发送选购需求"]')!.click());
    await act(async()=>host.querySelector<HTMLButtonElement>('.product-views .detail-button')!.click());
    await act(async()=>button('准备下单').click());
    expect(host.querySelector('.order-intent-product')!.textContent).toContain('213.99');
    await input('input[name="recipient_name"]','测试收件人');await input('input[name="city"]','测试城市');
    await input('textarea[name="address_line"]','测试地址');
    await act(async()=>button('生成确认单').click());
    expect(requests.find(r=>r.url.endsWith('/confirmations/orders'))!.body).toMatchObject({currency:'CNY',items:[{product_id:'P3021',sku_id:'P3021-S2',quantity:1}]});
    expect(orderCount).toBe(0);expect(host.textContent).toContain('238.99');
    await act(async()=>button('确认创建意向单').click());
    expect(orderCount).toBe(1);expect(host.textContent).toContain('GBX-view');
    expect(host.querySelectorAll('.product-card[data-card-mode="view"]')).toHaveLength(1);
    expect(products[1].currency).toBe('USD');
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});
