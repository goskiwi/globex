import {presented,shown} from './presentationFixture';
// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import App from './WorkspaceFixture';
import {renderToStaticMarkup} from 'react-dom/server';
import ResearchReports from '../src/components/ResearchReports';
import type {ResearchDelivery} from '../src/types';
import {readResearchReports} from '../src/lib/research';

const evidenceValue='完整目录原文仍保留，防泼水且可折叠。';
const report={task_id:'bag',title:'通勤背包研究',status:'partial',product_count:1,sku_count:1,
  caveats:['目录未标明背板结构'],unmet_requirements:['电脑仓资料尚未核实'],questions:['需要放多大电脑？'],
  issues:['补充来源暂不可用'],transaction_state:'awaiting_confirmation',price_comparisons:[],metric_comparisons:[],items:[{
    product_id:'P1003',sku_id:'P1003-S1',evidence_fields:['description','weight_kg','package_dimensions_cm'],
    evidence:[{field:'description',label:'目录描述',value:evidenceValue},{field:'weight_kg',label:'重量（kg）',value:0.38}],
    missing_fields:[{field:'package_dimensions_cm',label:'包装尺寸（cm）'}],
    facts:{product:{product_id:'P1003',title:'Wanderlite 折叠旅行双肩包 35L',weight_kg:0.38,
      description:'完整目录原文仍保留，防泼水且可折叠。',highlights:['支持折叠'],
      dimensions_cm:{length:50},material_tags:['尼龙'],rating_summary:{average:4.6,review_count:217},ships_to:['CN']},
      sku:{sku_id:'P1003-S1',spec:'石墨黑',price_major:11958.33,currency:'JPY',display_price_major:574,display_currency:'CNY',stock:5},quotes:[],historical:false}}]};
const delivery={runId:'r',guidance:'已交付1份研究报告。',reports:[report],status:'partial',task_ids:['bag'],completed_task_ids:[]};

it('研究组件完整展示缺口、问题、执行事项与确认状态，每项只出现一次',()=>{
  const html=renderToStaticMarkup(<ResearchReports delivery={delivery as ResearchDelivery}/>);
  for(const text of [report.title,'目录未标明背板结构','电脑仓资料尚未核实','需要放多大电脑？','补充来源暂不可用',
    '已有确认单等待页面批准，尚未执行交易。']){
    const visible=html.replace(/<[^>]*>/g,'');
    expect(visible.split(text)).toHaveLength(2);
  }
});

it('只读取当前结构化研究，不从旧正文构造候选',()=>{
  expect(readResearchReports([delivery])[0].reports[0].items[0].evidence[0].value).toBe(evidenceValue);
  expect(readResearchReports([{runId:'r',summary:'旧报告',candidates:[]}])).toEqual([]);
  const {evidence,evidence_fields,missing_fields,...old}=report.items[0];
  expect(readResearchReports([{...delivery,reports:[{...report,items:[{...old,assessment:'旧评语',caveats:[]}]}]}])).toEqual([]);
  expect(readResearchReports([{...delivery,reports:[{...report,items:[{product_id:'P1003'}]}]}])).toEqual([]);
});

it('结构化证据完整可读，资料可展开，无购物动作，追问刷新后仍归属原回复',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  let firstRun='',lastRun='',session='',messages:any[]=[],saved:any=null,posts=0;
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/ag-ui/run')){
      const body=JSON.parse(String(init?.body));lastRun=body.runId;session=body.threadId;posts++;
      if(posts===1){firstRun=lastRun;saved={...delivery,runId:firstRun};}
      messages=[...body.messages,{id:`${lastRun}:final:answer`,role:'assistant',content:posts===1?delivery.guidance:'继续解释，不重复整份研究。'}];
      const events=[{type:'RUN_STARTED',threadId:session,runId:lastRun},{type:'MESSAGES_SNAPSHOT',messages},
        {type:'STATE_SNAPSHOT',snapshot:{presentations:[...presented('recommendation',null,body.runId),...presented('comparison',null,body.runId),...shown('research',posts===1?[saved]:[])]}},
        {type:'RUN_FINISHED',threadId:session,runId:lastRun}];
      return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:session?[{id:session,title:'通勤研究',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,run:{runId:lastRun,threadId:session,status:'completed',messages,state:{}},presentations:[...shown('research',[saved]),...shown('product_view',[])]});
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
    expect(section.querySelector('.research-analysis')?.textContent).toContain(evidenceValue);
    const cards=section.querySelector('.product-grid')!;
    const analysis=section.querySelector('.research-analysis')!;
    expect(cards.compareDocumentPosition(analysis) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
    const turn=section.closest('.assistant-turn')!;
    expect(turn.querySelectorAll('h3')).toHaveLength(1);
    for(const text of [report.title, ...report.caveats, ...report.unmet_requirements, ...report.questions, ...report.issues,
      '已有确认单等待页面批准，尚未执行交易。']){
      expect(turn.textContent!.split(text)).toHaveLength(2);
      expect(turn.querySelector('.assistant-text')?.textContent).not.toContain(text);
    }
    expect(section.querySelector('.product-card')?.textContent).not.toContain(evidenceValue);
    expect(section.textContent).toContain('¥574.00');
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

it('跨币种研究卡复用读取时显示金额，保留已有的同规格到手报价',()=>{
 const data=structuredClone(delivery) as ResearchDelivery;
 const c=data.reports[0].items[0];
 c.facts.sku={sku_id:c.sku_id!,spec:'Grey',stock:30,price_major:54.94,currency:'SGD',display_price_major:291.18,display_currency:'CNY'};
 c.facts.quotes=[{ship_to:'CN',currency:'CNY',historical:false,line:{product_id:c.product_id,sku_id:c.sku_id!,title:'合成跨币种报价',quantity:1,
  unit_price_minor:29118,source_unit_price_minor:5494,source_currency:'SGD',currency:'CNY',subtotal_minor:29118,freight_minor:2500,tariff_minor:0,total_amount_minor:31618}}];
 const html=renderToStaticMarkup(<ResearchReports delivery={data}/>);
 expect(html).toContain('¥291.18');expect(html).toContain('¥316.18');
 expect(html).not.toContain('到手价待目的地与规格确认');
});

it('研究展示只消费Main交付的标题，不从旧goal或内部任务拼标题',()=>{
  const read=readResearchReports([delivery]);
  const html=renderToStaticMarkup(<ResearchReports delivery={read[0]}/>);
  expect(html).toContain('<h3>通勤背包研究</h3>');
  const {title,...withoutTitle}=report;
  expect(readResearchReports([{...delivery,reports:[{...withoutTitle,goal:'内部任务 description/highlights'}]}])).toEqual([]);
  expect(readResearchReports([{...delivery,reports:[{...report,title:'研究'.repeat(41)}]}])).toEqual([]);
  expect(readResearchReports([{...delivery,reports:[{...report,title:'🎒'.repeat(80)}]}])).toHaveLength(1);
});


it('参数排名按程序结果显示并列最高和缺失范围，不生成自由评语',()=>{
 const data=structuredClone(delivery) as ResearchDelivery;
 data.reports[0].metric_comparisons=[{field:'review_count',label:'目录评价数',unit:'条',compared_product_count:2,excluded_products:[{product_id:'P3000',reason:'missing_value'}],minimum:287,maximum:349,
  lowest:[{product_id:'P1001',title:'候选甲',value:287}],highest:[{product_id:'P1002',title:'候选乙',value:349}]}];
 const html=renderToStaticMarkup(<ResearchReports delivery={data}/>);
 expect(html).toContain('349');expect(html).toContain('候选乙');expect(html).toContain('P3000：当前资料缺少该数值');
 expect(html).toContain('所选证据未提供：包装尺寸');expect(html).not.toContain('assessment');
});

it('不足两条时仍展示具体排除原因，不把来源冲突显示成资料缺失',()=>{
 const data=structuredClone(delivery) as ResearchDelivery;
 data.reports[0].metric_comparisons=[{field:'weight_kg',label:'重量',unit:'kg',compared_product_count:1,
  excluded_products:[{product_id:'P1003',reason:'conflicting_observations'},{product_id:'P3000',reason:'historical_only'}],
  minimum:null,maximum:null,lowest:[],highest:[]}];
 const accepted=readResearchReports([data]);expect(accepted).toHaveLength(1);
 const html=renderToStaticMarkup(<ResearchReports delivery={accepted[0]}/>);
 expect(html).toContain('不足两条可比较记录，未生成排名');
 expect(html).toContain('当前来源数值冲突，无法确定');expect(html).toContain('只有历史资料，尚未重新核验');
 expect(html).not.toContain('<dt>最低</dt>');expect(html).not.toContain('<dt>最高</dt>');
});
