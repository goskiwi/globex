import {presented,shown} from './presentationFixture';
// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import App from './WorkspaceFixture';

it('查看卡进入同一详情抽屉，全部平台规格可选，追问刷新留原回复并保留购买入口',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  const products=['amazon','ebay'].map((source_platform,index)=>({product_id:`P302${index}`,title:'Roamix 航空颈枕',
    brand:'Roamix',category:'旅行装备',origin_country:'US',source_platform,canonical_product_id:'CAN-GX-02-2',price_major:index?213.99:198,currency:'CNY',
    description:'单侧支撑，可拆洗纯棉外套。',highlights:[],score:1,default_sku_id:`P302${index}-S1`,
    skus:[1,2].map(n=>({sku_id:`P302${index}-S${n}`,spec:n===1?'石墨黑':'雾灰',
      price_major:index?30.14:198,currency:index?'USD':'CNY',stock:index&&n===2?95:0,
      display_price_major:index?213.99:198,display_currency:'CNY',constraint_issues:['over_price_cap']}))}));
  let firstRun='',lastRun='',session='',messages:any[]=[],view:any=null,posts=0;
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/ag-ui/run')){
      const body=JSON.parse(String(init?.body));lastRun=body.runId;session=body.threadId;posts++;
      const text=posts===1?'下面是两条平台记录的详情。':'后续解释，不修改预算。';
      messages=[...body.messages,{id:`${body.runId}:final:answer`,role:'assistant',content:text}];
      if(posts===1){firstRun=body.runId;view={runId:firstRun,guidance:text,hits:products,result_ref:'ctx_view',purpose:'product_view'};}
      const events=[{type:'RUN_STARTED',threadId:session,runId:lastRun},
        {type:'MESSAGES_SNAPSHOT',messages},{type:'STATE_SNAPSHOT',snapshot:{presentations:[...presented('recommendation',null,body.runId),...presented('comparison',null,body.runId),...shown('product_view',posts===1?[view]:[])]}},
        {type:'RUN_FINISHED',threadId:session,runId:lastRun}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:session?[{id:session,title:'查看颈枕',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,run:{runId:lastRun,threadId:session,status:'completed',messages,state:{}},presentations:[...shown('product_view',[view])]});
    return Response.json({sessions:[],confirmations:[],skills:[],products:[],form:null,revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);let root=createRoot(host);
  async function ask(text:string){await act(async()=>{
    const input=host.querySelector<HTMLTextAreaElement>('#query')!;
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(input,text);
    input.dispatchEvent(new Event('input',{bubbles:true}));
  });await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="发送选购需求"]')!.click());}
  const verify=async()=>{
    expect(host.querySelectorAll('.product-card[data-card-mode="view"]')).toHaveLength(1);
    expect(host.querySelector('.product-offers')).toBeNull();
    expect(host.textContent).toContain('213.99');
    expect(host.querySelector('.product-views')?.closest('.assistant-turn')?.getAttribute('data-message-id')).toBe(`${firstRun}:final:answer`);
    expect(host.querySelector('.search-results')).toBeNull();
    await act(async()=>host.querySelector<HTMLButtonElement>('.product-views .detail-button')!.click());
    const drawer=host.querySelector('.drawer')!;
    expect(drawer.querySelectorAll('input[name="platform"]')).toHaveLength(2);
    expect(drawer.querySelectorAll('input[name="product-sku"]')).toHaveLength(2);
    expect(drawer.textContent).toContain('30.14');expect(drawer.textContent).toContain('超出当前选购预算');
    expect([...drawer.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent==='准备下单')!.disabled).toBe(false);
    await act(async()=>drawer.querySelector<HTMLInputElement>('input[name="platform"][value="P3020"]')!.click());
    expect(drawer.querySelectorAll('input[name="product-sku"]')).toHaveLength(2);
    expect(drawer.textContent).toContain('缺货');
    expect([...drawer.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent==='当前规格缺货')!.disabled).toBe(true);
    await act(async()=>drawer.querySelector<HTMLButtonElement>('[aria-label="关闭"]')!.click());
  };
  try{
    await act(async()=>root.render(<App/>));await ask('看看这两款');await verify();
    await ask('解释一下');await verify();
    await act(async()=>root.unmount());root=createRoot(host);await act(async()=>root.render(<App/>));await verify();
    expect(posts).toBe(2);
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});

it('查看指定非默认规格时金额与抽屉起点保持同一SKU，全部规格仍可查看',async()=>{
  const {default:ProductCards}=await import('../src/components/ProductCards');
  const {initialOffer}=await import('../src/lib/productGroups');
  const {readProducts}=await import('../src/lib/commerceClient');
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  let opened:any;
  const products=readProducts([{product_id:'P3004',title:'查看商品',brand:'合成',score:1,category:'旅行装备',origin_country:'CN',currency:'CNY',price_major:107.99,
    default_sku_id:'P3004-S2',requested_sku_ids:['P3004-S2'],highlights:[],skus:[{sku_id:'P3004-S1',spec:'原规格',price_major:98.97,currency:'CNY',stock:10},{sku_id:'P3004-S2',spec:'指定规格',price_major:107.99,currency:'CNY',stock:10}]}]);
  try{
    await act(async()=>root.render(<ProductCards mode="view" products={products} onDetail={g=>{opened=g;}}/>));
    expect(host.textContent).toContain('107.99');expect(host.textContent).toContain('指定规格');expect(host.textContent).not.toContain('98.97');
    await act(async()=>host.querySelector<HTMLButtonElement>('.detail-button')!.click());
    expect(initialOffer(opened)?.sku.sku_id).toBe('P3004-S2');expect(opened.records[0].skus).toHaveLength(2);
  }finally{await act(async()=>root.unmount());host.remove();}
});

it('查看卡与抽屉使用同一匹配规格报价，不因只读模式丢失已核实到手价',async()=>{
  const {renderToStaticMarkup}=await import('react-dom/server');
  const {default:ProductCards}=await import('../src/components/ProductCards');
  const base={product_id:'P3004',title:'指定规格',brand:'测试',category:'旅行装备',origin_country:'CN',
    price_major:107.99,currency:'CNY',highlights:[],score:1,default_sku_id:'P3004-S2',requested_sku_ids:['P3004-S2','P3004-S1'],
    skus:[{sku_id:'P3004-S1',spec:'石墨黑',price_major:98.97,currency:'CNY',stock:2},
      {sku_id:'P3004-S2',spec:'雾灰',price_major:107.99,currency:'CNY',stock:3}],
    landed_price:{ship_to:'CN',currency:'CNY',subtotal_minor:10799,freight_minor:2500,tariff_minor:0,total_amount_minor:13299,
      items:[{product_id:'P3004',sku_id:'P3004-S2',title:'指定规格',quantity:1,unit_price_minor:10799,currency:'CNY',
        source_unit_price_minor:10799,source_currency:'CNY',subtotal_minor:10799,freight_minor:2500,tariff_minor:0,total_amount_minor:13299}]}};
  const matching=renderToStaticMarkup(<ProductCards mode="view" products={[base]} onDetail={()=>{}}/>);
  expect(matching).toContain('132.99');expect(matching).toContain('配送至');expect(matching).not.toContain('到手价待目的地与规格确认');
  const line=base.landed_price.items[0];
  for(const wrong of [
    {...base,landed_price:undefined},
    {...base,landed_price:{...base.landed_price,items:[{...line,sku_id:'P3004-S1'}]}},
    {...base,landed_price:{...base.landed_price,items:[{...line,quantity:2}]}},
    {...base,landed_price:{...base.landed_price,currency:'USD'}},
  ]){
    const html=renderToStaticMarkup(<ProductCards mode="view" products={[wrong]} onDetail={()=>{}}/>);
    expect(html).toContain('到手价待目的地与规格确认');expect(html).not.toContain('132.99');
  }
});
