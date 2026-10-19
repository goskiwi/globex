import {presented,shown} from './presentationFixture';
// @vitest-environment jsdom
import {act} from "react";
import {createRoot} from "react-dom/client";
import {expect,it,vi} from "vitest";
import App from "./WorkspaceFixture";
import {readComparison,readRecommendation} from "../src/lib/commerceClient";
import type {ProductCard} from "../src/types";

const products: ProductCard[] = [1,2,3,4].map(n=>({
  product_id:`P900${n}`,default_sku_id:`P900${n}-S1`,title:`合成背包${n}`,
  brand:"test",category:"旅行装备",origin_country:"CN",price_major:100,currency:"CNY",
  highlights:[],score:1,quantity:1,skus:[{sku_id:`P900${n}-S1`,spec:"黑色",price_major:100,currency:"CNY",stock:10}],
  recommendation_reason:"用户用途对应的理由",tradeoffs:["隔层资料未提供"],
}));
const decision={guidance:"通勤时优先考虑收纳和背负，按这次携带物品选择首选。".repeat(14),hits:products,preferred_sku_id:"P9002-S1",dimensions:["收纳","背负"],max_items:12,
  result_ref:"ctx_test",unverified_requirements:[]};

it("推荐只承载选择和说明，比较保留自己的倾向、关注点与容量",()=>{
  expect(readComparison(decision)?.hits).toHaveLength(4);
  expect(readComparison(decision)?.dimensions).toEqual(['收纳','背负']);
  const recommendation={guidance:decision.guidance,hits:products,result_ref:'ctx_test',unverified_requirements:[],mode:'individual',quote:null};
  const read=readRecommendation(recommendation)!;
  expect(read.hits).toHaveLength(4);
  expect(read).not.toHaveProperty('dimensions');expect(read).not.toHaveProperty('preferred_sku_id');
  for(const key of ['guidance','hits','result_ref']){
    const missing={...recommendation} as Record<string,unknown>;delete missing[key];expect(readRecommendation(missing)).toBeNull();
  }
  for(const key of ['preferred_sku_id','dimensions','max_items']){
    const missing={...decision} as Record<string,unknown>;delete missing[key];expect(readComparison(missing)).toBeNull();
  }
  expect(readComparison({...decision,preferred_sku_id:'P9999-S1'})).toBeNull();
  expect(readComparison({...decision,max_items:3})).toBeNull();
  expect(readRecommendation({...recommendation,guidance:'  '})).toBeNull();
});

it("推荐卡片绑定原回复，追问与刷新不移动，新的比较在新回复下交付",async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();
  vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  let posts=0,firstRun='',lastRun='',sessionId='',messages:any[]=[],state:any={},records:any[]=[];
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/ag-ui/run')){
      const body=JSON.parse(String(init?.body));posts++;lastRun=body.runId;sessionId=body.threadId;
      if(posts===1)firstRun=body.runId;
      const text=posts===1?'首轮选购建议':posts===2?'这是后续查库存的回答':'新的比较建议';
      messages=[...body.messages,{id:`${body.runId}:final:answer`,role:'assistant',content:text}];
      if(posts===1)state={presentations:[...presented('recommendation',{...decision,guidance:text,mode:'individual',quote:null},body.runId),...presented('comparison',null,body.runId)]};
      if(posts===3)state={presentations:[...presented('recommendation',null,body.runId),...presented('comparison',{...decision,guidance:text},body.runId)]};
      for(const p of state.presentations??[])if(!records.some(r=>r.id===p.id))records.push(p);
      const events=[{type:'RUN_STARTED',threadId:body.threadId,runId:body.runId},
        {type:'MESSAGES_SNAPSHOT',messages},{type:'STATE_SNAPSHOT',snapshot:state},
        {type:'RUN_FINISHED',threadId:body.threadId,runId:body.runId}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:sessionId?[{id:sessionId,title:'测试选购',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,presentations:records,events:[],
      run:{runId:lastRun,threadId:sessionId,status:'completed',messages,state}});
    return Response.json({sessions:[],confirmations:[],skills:[],products:[],form:null,revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);let root=createRoot(host);
  async function ask(query:string){
    await act(async()=>{
      const input=host.querySelector<HTMLTextAreaElement>('#query')!;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(input,query);
      input.dispatchEvent(new Event('input',{bubbles:true}));
    });
    await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="发送选购需求"]')!.click());
  }
  const turn=(run:string)=>host.querySelector<HTMLElement>(`[data-message-id="${run}:final:answer"]`)!;
  try{
    await act(async()=>root.render(<App/>));
    await ask('推荐背包');await ask('核对库存');
    expect(turn(firstRun).querySelectorAll('.product-card')).toHaveLength(4);
    expect(turn(lastRun).querySelector('.search-results')).toBeNull();
    expect(host.querySelectorAll('.search-results')).toHaveLength(1);
    expect(turn(firstRun).compareDocumentPosition(turn(lastRun)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    await act(async()=>root.unmount());root=createRoot(host);
    await act(async()=>root.render(<App/>));
    expect(posts).toBe(2);
    expect(turn(firstRun).querySelectorAll('.product-card')).toHaveLength(4);
    expect(turn(lastRun).querySelector('.search-results')).toBeNull();
    await ask('比较这几款');
    expect(turn(firstRun).querySelector('.historical-products')).not.toBeNull();
    expect(turn(firstRun).querySelector('.search-results')).toBeNull();
    expect(turn(lastRun).querySelectorAll('.product-card')).toHaveLength(4);
    expect(turn(lastRun).querySelector('.compare-table')).toBeNull();
    await act(async()=>[...turn(lastRun).querySelectorAll<HTMLButtonElement>('button')].find(button=>button.textContent==='查看对比')!.click());
    expect(host.querySelector('.modal .compare-table')).not.toBeNull();
    expect(host.querySelectorAll('.search-results')).toHaveLength(1);
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});

it("实际 App 推荐卡可勾选比较，不自动附加比较倾向与维度，不重新请求模型",async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();
  vi.stubGlobal("scrollTo",vi.fn());Element.prototype.scrollIntoView=vi.fn();
  const requests:string[]=[];
  vi.stubGlobal("fetch",vi.fn(async(url:string,init?:RequestInit)=>{
    requests.push(String(url));
    if(String(url).endsWith("/ag-ui/run")) {
      const body=JSON.parse(String(init?.body));
      const events=[{type:"RUN_STARTED",threadId:body.threadId,runId:body.runId},
        {type:"STATE_SNAPSHOT",snapshot:{skillUsages:[],progress:[],presentations:[...presented('recommendation',{...decision,mode:'individual',quote:null},body.runId),...presented('comparison',null,body.runId)]}},
        {type:"MESSAGES_SNAPSHOT",messages:[...body.messages,{id:`${body.runId}:final:answer`,role:"assistant",content:decision.guidance}]},
        {type:"RUN_FINISHED",threadId:body.threadId,runId:body.runId}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(""),{headers:{"Content-Type":"text/event-stream"}});
    }
    return Response.json({sessions:[],confirmations:[],skills:[],products:[],form:null,revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  try {
    await act(async()=>root.render(<App/>));
    expect(host.querySelector('.sidebar-note')).toBeNull();
    expect(host.querySelector('.little-orbit')).toBeNull();
    expect(host.querySelector('.sidebar-history .sidebar-history-list')).not.toBeNull();
    expect(host.querySelector('.sidebar-bottom .profile')).not.toBeNull();
    await act(async()=>{
      const input=host.querySelector<HTMLTextAreaElement>('#query')!;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(input,'比较通勤背包');
      input.dispatchEvent(new Event('input',{bubbles:true}));
    });
    await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="发送选购需求"]')!.click());
    expect(host.querySelectorAll('.preferred-pick')).toHaveLength(0);
    expect(host.querySelector('.composer-plan-toggle')).toBeNull();
    expect(host.textContent).not.toContain('结果来自商品目录');
    expect(host.textContent).toContain(decision.guidance);
    expect(host.querySelector('.answer-collapsible')).toBeNull();
    const modelCalls=requests.filter(u=>u.endsWith('/ag-ui/run')).length;
    const quick=[...host.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent?.includes('一起比较看看'))!;
    await act(async()=>quick.click());
    expect(document.querySelector('.compare-table')!.querySelectorAll('thead th')).toHaveLength(5);
    await act(async()=>host.querySelector<HTMLButtonElement>('[role="dialog"] [aria-label="关闭"]')!.click());
    await act(async()=>host.querySelector<HTMLButtonElement>('.compare-clear')!.click());
    for(const checkbox of host.querySelectorAll<HTMLInputElement>('.compare-check input'))
      await act(async()=>checkbox.click());
    expect(host.textContent).toContain('已选 4 件');
    await act(async()=>host.querySelector<HTMLButtonElement>('.compare-go')!.click());
    const table=document.querySelector('.compare-table')!;
    expect(table.querySelectorAll('thead th')).toHaveLength(5);
    expect(document.body.textContent).not.toContain('本次关注：收纳、背负');
    expect(table.textContent).not.toContain('更推荐这款');
    expect(table.textContent).toContain('隔层资料未提供');
    expect(requests.filter(u=>u.endsWith('/ag-ui/run'))).toHaveLength(modelCalls);
  } finally {
    await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();
  }
});
