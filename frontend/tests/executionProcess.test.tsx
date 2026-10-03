// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import App from './WorkspaceFixture';
import {CommerceClient} from './clientFixture';
import {readProcess,upsertProcess} from '../src/lib/execution';
import ExecutionProcess from '../src/components/ExecutionProcess';

it('参数拒绝和后续成功独立显示，不把明确未执行写成结果未确认',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  const process={runId:'correction',userMessageId:'u',status:'completed' as const,steps:[
    {id:'rejected',label:'准备最终推荐',status:'failed' as const,summary:'参数校验未通过：picks.1.quantity：缺少必填字段；本次未执行'},
    {id:'delivered',label:'交付最终推荐',status:'completed' as const,summary:'已交付2款推荐商品；首选：合成颈枕'}]};
  try{
    await act(async()=>root.render(<ExecutionProcess process={process}/>));
    await act(async()=>host.querySelector<HTMLButtonElement>('.execution-heading')!.click());
    const rows=host.querySelectorAll('.execution-step');
    expect(rows).toHaveLength(2);expect(rows[0].textContent).toContain('缺少必填字段');
    expect(rows[0].textContent).toContain('本次未执行');expect(rows[0].textContent).not.toContain('未确认');
    expect(rows[1].textContent).toContain('已交付2款');
  }finally{await act(async()=>root.unmount());host.remove();}
});

it('流程只读取新结构，重复同一调用更新一行，同名不同ID独立保留',()=>{
  expect(readProcess({progress:[{label:'成功'}]})).toBeNull();
  const p=readProcess({runId:'r',userMessageId:'u',status:'running',steps:[
    {id:'a',label:'检索商品',status:'running',summary:''},
    {id:'b',label:'检索商品',status:'completed',summary:'返回2个候选'},
    {id:'a',label:'检索商品',status:'completed',summary:'返回0个候选'}]})!;
  expect(p.steps).toHaveLength(2);expect(p.steps[0].summary).toContain('0个');
  expect(upsertProcess([p],{...p,status:'completed'})).toHaveLength(1);
});

it('真实App结果前出现动作和摘要，终态保留原用户消息下，追问和刷新不串轮',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  let firstRun='',session='',messages:any[]=[],processes:any[]=[],controller:ReadableStreamDefaultController|undefined,posts=0;
  const send=(event:any)=>controller!.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/ag-ui/run')){
      const body=JSON.parse(String(init?.body));posts++;session=body.threadId;
      const p={runId:body.runId,userMessageId:body.messages.at(-1).id,status:'running',steps:[
        {id:body.runId+':a',label:'更新选购条件',status:'completed',summary:'到手预算300 CNY；配送至中国'},
        {id:body.runId+':b',label:'检索商品',status:'running',summary:''}]};
      processes.push(p);messages=body.messages;
      if(posts===1){
        firstRun=body.runId;
        return new Response(new ReadableStream({start(c){controller=c;send({type:'RUN_STARTED',runId:body.runId,threadId:session});
          send({type:'STATE_SNAPSHOT',snapshot:{process:p}});}}),{headers:{'Content-Type':'text/event-stream'}});
      }
      p.status='completed';p.steps[1]={...p.steps[1],status:'completed',summary:'返回1个候选'};
      messages=[...body.messages,{id:body.runId+':final:answer',role:'assistant',content:'后续回答'}];
      return new Response([{type:'RUN_STARTED',runId:body.runId,threadId:session},
        {type:'STATE_SNAPSHOT',snapshot:{process:p}},{type:'MESSAGES_SNAPSHOT',messages},
        {type:'RUN_FINISHED',runId:body.runId,threadId:session}].map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
    }
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:session?[{id:session,title:'合成购物',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,processes,productHistory:[],productViews:[],events:[],
      run:{runId:processes.at(-1).runId,threadId:session,status:'completed',messages,state:{process:processes.at(-1)}}});
    return Response.json({sessions:[],confirmations:[],skills:[],products:[],form:null,revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);let root=createRoot(host);
  const process=()=>host.querySelector<HTMLElement>(`[data-run-id="${firstRun}"]`)!;
  async function ask(text:string){await act(async()=>{
    const input=host.querySelector<HTMLTextAreaElement>('#query')!;
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(input,text);
    input.dispatchEvent(new Event('input',{bubbles:true}));
  });await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="发送选购需求"]')!.click());}
  try{
    await act(async()=>root.render(<App/>));await ask('通勤背包300元寄中国');
    expect(process()?.textContent).toContain('检索商品');
    expect(process()?.querySelectorAll('.execution-step')).toHaveLength(0);
    await act(async()=>process().querySelector<HTMLButtonElement>('.execution-heading')!.click());
    await vi.waitFor(()=>expect(process()?.textContent).toContain('配送至中国'));
    expect(process().querySelectorAll('.execution-step')).toHaveLength(2);
    expect(process().textContent).toContain('检索商品');expect(host.querySelector('.product-card')).toBeNull();
    expect(process().closest('.user-turn')!.querySelector('.query-bubble')!.textContent).toContain('通勤背包');
    await act(async()=>{
      processes[0]={...processes[0],status:'completed',steps:processes[0].steps.map((s:any)=>s.id.endsWith(':b')?{...s,status:'completed',summary:'返回3个候选'}:s)};
      messages=[...messages,{id:firstRun+':final:answer',role:'assistant',content:'最终建议'}];
      send({type:'STATE_SNAPSHOT',snapshot:{process:processes[0]}});send({type:'MESSAGES_SNAPSHOT',messages});
      send({type:'RUN_FINISHED',runId:firstRun,threadId:session});controller!.close();
    });
    await act(async()=>process().querySelector<HTMLButtonElement>('.execution-heading')!.click());
    expect(process().textContent).toContain('返回3个候选');expect(process().querySelectorAll('.execution-step')).toHaveLength(2);
    await ask('核对一下');
    expect(host.querySelectorAll('.execution-process')).toHaveLength(2);
    expect(process().textContent).toContain('返回3个候选');
    await act(async()=>root.unmount());root=createRoot(host);await act(async()=>root.render(<App/>));
    await vi.waitFor(()=>expect(host.querySelectorAll('.execution-process')).toHaveLength(2));
    await act(async()=>process().querySelector<HTMLButtonElement>('.execution-heading')!.click());
    expect(process().textContent).toContain('返回3个候选');
    expect(host.querySelector('.progress-line')).toBeNull();expect(host.querySelector('.skill-run-status')).toBeNull();
    expect(posts).toBe(2);
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});

it('SDK只认取消终态，保留之前完成的步骤；迟到旧轮不覆盖新会话',async()=>{
  const client=new CommerceClient({url:'/run',fetch:async(_,init)=>{
    const body=JSON.parse(String(init.body));const p={runId:body.runId,userMessageId:body.messages.at(-1).id,status:'cancelled',steps:[
      {id:'a',label:'检索商品',status:'completed',summary:'返回2个候选'},
      {id:'b',label:'核对报价',status:'cancelled',summary:'调用已停止，未取得完整回执'}]};
    return new Response([{type:'RUN_STARTED',threadId:body.threadId,runId:body.runId},
      {type:'STATE_SNAPSHOT',snapshot:{process:p}},{type:'RUN_ERROR',code:'CANCELLED',message:'已停止'}]
      .map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
  }});
  await client.submit('购物');expect(client.getSnapshot().processes[0].status).toBe('cancelled');
  expect(client.getSnapshot().processes[0].steps[0].status).toBe('completed');
  client.reset();expect(client.getSnapshot().processes).toEqual([]);
});
