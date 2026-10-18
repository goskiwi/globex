// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import App from './WorkspaceFixture';
import {renderToStaticMarkup} from 'react-dom/server';
import ResearchReports from '../src/components/ResearchReports';
import type {ResearchDelivery} from '../src/types';
import {readResearchReports} from '../src/lib/research';

const assessment='适合优先减轻随身负担的人，折叠后也方便收纳。若每天携带电脑，独立电脑仓和保护能力仍需核验。';
const report={task_id:'bag',goal:'通勤背包研究',status:'partial',product_count:1,sku_count:1,
  unmet_constraints:['电脑仓资料尚未核实'],questions:[],issues:[],price_comparisons:[],items:[{
    product_id:'P1003',sku_id:'P1003-S1',assessment,open_questions:['是否有独立电脑仓？'],unmet_constraints:[],
    facts:{product:{product_id:'P1003',title:'Wanderlite 折叠旅行双肩包 35L',weight_kg:0.38,
      description:'完整目录原文仍保留，防泼水且可折叠。',highlights:['支持折叠'],
      dimensions_cm:{length:50},material_tags:['尼龙'],rating_summary:{average:4.6,review_count:217},ships_to:['CN']},
      sku:{sku_id:'P1003-S1',spec:'石墨黑',price_major:11958.33,currency:'JPY',stock:5},quotes:[],historical:false}}]};
const delivery={runId:'r',guidance:'背包研究已交回，电脑仓资料尚未核实。',reports:[report],status:'partial',task_ids:['bag'],completed_task_ids:[]};

it('只读取当前结构化研究，不从旧正文构造候选',()=>{
  expect(readResearchReports([delivery])[0].reports[0].items[0].assessment).toBe(assessment);
  expect(readResearchReports([{runId:'r',summary:'旧报告',candidates:[]}])).toEqual([]);
  expect(readResearchReports([{...delivery,reports:[{...report,items:[{product_id:'P1003'}]}]}])).toEqual([]);
});

it('研究取舍完整可读，资料可展开，无购物动作，追问刷新后仍归属原回复',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  let firstRun='',lastRun='',session='',messages:any[]=[],saved:any=null,posts=0;
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/ag-ui/run')){
      const body=JSON.parse(String(init?.body));lastRun=body.runId;session=body.threadId;posts++;
      if(posts===1){firstRun=lastRun;saved={...delivery,runId:firstRun};}
      messages=[...body.messages,{id:`${lastRun}:final:answer`,role:'assistant',content:posts===1?delivery.guidance:'继续解释，不重复整份研究。'}];
      const events=[{type:'RUN_STARTED',threadId:session,runId:lastRun},{type:'MESSAGES_SNAPSHOT',messages},
        {type:'STATE_SNAPSHOT',snapshot:{researchReports:posts===1?[saved]:[],recommendation:null,comparison:null}},
        {type:'RUN_FINISHED',threadId:session,runId:lastRun}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:session?[{id:session,title:'通勤研究',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,researchReports:[saved],productViews:[],productHistory:[],
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
    const section=host.querySelector<HTMLElement>('.research-reports')!;
    expect(section).not.toBeNull();
    expect(section.closest('.assistant-turn')?.getAttribute('data-message-id')).toBe(`${firstRun}:final:answer`);
    expect(section.querySelector('.research-analysis')?.textContent).toContain(assessment);
    expect(section.querySelector('.product-card')?.textContent).not.toContain(assessment);
    expect(section.textContent).toContain('11,958.33');
    expect(section.textContent).toContain('部分完成');
    expect(section.querySelector('.answer-collapsible')).toBeNull();
    expect(section.querySelectorAll('.product-card[data-card-mode="research"]')).toHaveLength(1);
    expect(section.querySelector('.product-visual')).not.toBeNull();
    expect(section.querySelector('.research-source')).toBeNull();
    await act(async()=>section.querySelector<HTMLButtonElement>('.detail-button')!.click());
    const drawer=host.querySelector('.drawer')!;
    expect(drawer.textContent).toContain('完整目录原文仍保留');
    expect(drawer.textContent).toContain('重量：0.38 kg');
    for (const label of ['目录亮点：支持折叠','商品尺寸（厘米）：长 50','目录材质：尼龙','目录评分：评分 4.6 · 评价数 217','目录配送范围：CN']) {
      expect(drawer.textContent).toContain(label);
    }
    expect(drawer.textContent).not.toContain('准备下单');
    await act(async()=>drawer.querySelector<HTMLButtonElement>('[aria-label="关闭"]')!.click());
    expect(section.textContent).not.toContain('准备下单');
    expect(section.textContent).not.toContain('选这款');
  };
  try{
    await act(async()=>root.render(<App/>));await ask('研究背包');await verify();
    await ask('继续解释');await verify();
    await act(async()=>root.unmount());root=createRoot(host);await act(async()=>root.render(<App/>));await verify();
    expect(posts).toBe(2);
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});


it('研究卡片仅展示唯一匹配规格、数量和币种的报价，不猜多目的地报价',()=>{
  const data=structuredClone(delivery) as ResearchDelivery;
  const candidate=data.reports[0].items[0];
  candidate.facts.sku!.display_price_major=574;candidate.facts.sku!.display_currency='CNY';
  const quote={ship_to:'CN',currency:'CNY',historical:false,line:{product_id:'P1003',sku_id:'P1003-S1',title:'合成报价',quantity:1,
    unit_price_minor:57400,source_unit_price_minor:1195833,source_currency:'JPY',currency:'CNY',
    subtotal_minor:57400,freight_minor:2500,tariff_minor:0,total_amount_minor:59900}};
  candidate.facts.quotes=[quote];
  let html=renderToStaticMarkup(<ResearchReports delivery={data}/>);
  expect(html).toContain('¥574.00');expect(html).toContain('¥599.00');
  candidate.facts.quotes.push({...quote,ship_to:'US'});
  html=renderToStaticMarkup(<ResearchReports delivery={data}/>);
  expect(html).toContain('到手价待目的地与规格确认');expect(html).not.toContain('¥599.00');
  candidate.facts.quotes=[{...quote,line:{...quote.line,sku_id:'P1003-S2'}}];
  html=renderToStaticMarkup(<ResearchReports delivery={data}/>);
  expect(html).not.toContain('¥599.00');
  candidate.facts.quotes=[{...quote,line:{...quote.line,currency:'USD'}}];
  expect(renderToStaticMarkup(<ResearchReports delivery={data}/>)).not.toContain('¥599.00');
  candidate.facts.quotes=[{...quote,line:{...quote.line,quantity:2}}];
  expect(renderToStaticMarkup(<ResearchReports delivery={data}/>)).not.toContain('¥599.00');
});
