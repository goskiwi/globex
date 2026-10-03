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
        {type:'MESSAGES_SNAPSHOT',messages},{type:'STATE_SNAPSHOT',snapshot:{productViews:posts===1?[view]:[],recommendation:null,comparison:null}},
        {type:'RUN_FINISHED',threadId:session,runId:lastRun}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:session?[{id:session,title:'查看颈枕',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,productViews:[view],productHistory:[],
      run:{runId:lastRun,threadId:session,status:'completed',messages,state:{}}});
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
