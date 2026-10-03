// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import App from './WorkspaceFixture';

it('完整App表单归属原回复，提交后接一次结果、收起表单，刷新不复活旧待办',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  let session='',origin='',messages:any[]=[],processes:any[]=[],form:any=null,posts=0,writes=0,lastRun='';
  const question=(id:string,label:string)=>({id,type:'text',label,required:true,help_text:'',placeholder:'',unit:'',minimum:null,maximum:null,options:[]});
  function makeForm(){const id='form-owned';return {form_id:id,session_id:session,origin_run_id:origin,
    origin_message_id:messages[0].id,revision:1,submission:null,messages:[
      {version:'v0.9',createSurface:{surfaceId:id,catalogId:'globex.local/shopping-v2'}},
      {version:'v0.9',updateComponents:{surfaceId:id,components:[{id:'root',component:'ShoppingForm',title:'补充使用方式',context:'',description:'',
        questions:[question('scene','使用场景'),question('priority','更看重什么')],value:{path:'/requirements'},action:{event:{name:'applyShoppingRequirements'}}}]}},
      {version:'v0.9',updateDataModel:{surfaceId:id,path:'/',value:{requirements:{},revision:1}}}
    ]};}
  const product={product_id:'P1003',title:'轻便通勤背包',brand:'测试品牌',category:'旅行装备',origin_country:'CN',price_major:129,currency:'CNY',
    score:0,highlights:['可折叠'],default_sku_id:'P1003-S1',skus:[{sku_id:'P1003-S1',spec:'黑色',price_major:129,currency:'CNY',stock:50}],recommendation_reason:'适合随身携带'};
  const decision={mode:'alternatives',preferred_sku_id:'P1003-S1',dimensions:['便携收纳'],max_items:12,guidance:'更推荐便携款。',
    hits:[product],quote:null,unverified_requirements:[],result_ref:'ctx_owned'};
  const sse=(events:any[])=>new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}});
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/ag-ui/run')){
      const body=JSON.parse(String(init?.body));posts++;session=body.threadId;messages=body.messages;lastRun=body.runId;
      const process={runId:lastRun,userMessageId:messages.at(-1).id,status:posts===1?'waiting_input':'completed',steps:[]};processes.push(process);
      if(posts===1){origin=lastRun;form=makeForm();}
      messages=[...messages,{id:lastRun+':final:answer',role:'assistant',content:posts===1?'请补充使用方式。':'更推荐便携款。'}];
      const state=posts===1?{process,shoppingForms:[form]}:{process,recommendation:decision,deliveredRunId:lastRun,shoppingForms:[]};
      return sse([{type:'RUN_STARTED',runId:lastRun,threadId:session},{type:'STATE_SNAPSHOT',snapshot:state},
        {type:'MESSAGES_SNAPSHOT',messages},{type:'RUN_FINISHED',runId:lastRun,threadId:session}]);
    }
    if(url.includes('/shopping-forms/form-owned/actions')){
      writes++;const values=JSON.parse(String(init?.body)).action.context.requirements;
      const submission={query:'结构化答案正文，不用于UI解析',run_id:'form-run-'+'a'.repeat(32),status:'submitted',values};
      form={...form,revision:2,submission,messages:form.messages.map((m:any,i:number)=>i===2?{...m,updateDataModel:{...m.updateDataModel,value:{requirements:values,revision:2}}}:m)};
      return Response.json(submission);
    }
    if(url.includes('/shopping-forms?'))return Response.json({forms:form?[form]:[]});
    if(url.includes('/ag-ui/sessions?'))return Response.json({sessions:session?[{id:session,title:'测试选购',updatedAt:1}]:[]});
    if(url.includes('/ag-ui/sessions/'))return Response.json({messages,processes,productViews:[],productHistory:[],events:[],
      run:{runId:lastRun,threadId:session,status:'completed',messages,state:{recommendation:posts>1?decision:null,deliveredRunId:posts>1?lastRun:null,process:processes.at(-1)}}});
    return Response.json({forms:[],skills:[],confirmations:[],products:[],sessions:[],revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);let root=createRoot(host);
  async function fill(selector:string,value:string){await act(async()=>{
    const input=host.querySelector<HTMLInputElement|HTMLTextAreaElement>(selector)!;
    Object.getOwnPropertyDescriptor(input instanceof HTMLTextAreaElement?HTMLTextAreaElement.prototype:HTMLInputElement.prototype,'value')!.set!.call(input,value);
    input.dispatchEvent(new Event('input',{bubbles:true}));
  });}
  try{
    await act(async()=>root.render(<App/>));await fill('#query','帮我选一个背包');
    await act(async()=>host.querySelector<HTMLButtonElement>('[aria-label="发送选购需求"]')!.click());
    await vi.waitFor(()=>expect(host.querySelector('.shopping-form')).not.toBeNull());
    expect(host.querySelector('.shopping-form')!.closest('.assistant-turn')?.getAttribute('data-message-id')).toBe(origin+':final:answer');
    await fill('[name="scene"]','通勤');await fill('[name="priority"]','便携收纳');
    await act(async()=>host.querySelector<HTMLFormElement>('.shopping-form form')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
    await vi.waitFor(()=>expect(host.querySelector('.product-card')).not.toBeNull());
    expect(posts).toBe(2);expect(writes).toBe(1);
    expect(host.querySelector('.shopping-form')).toBeNull();expect(host.textContent).not.toContain('按已保存条件继续选购');
    expect(host.querySelector('.shopping-form-receipt button')).toBeNull();
    expect(host.querySelector(`.execution-process[data-run-id="${origin}"]`)?.textContent).toContain('已补充信息');
    expect(host.querySelector('.product-card')!.closest('.assistant-turn')?.getAttribute('data-message-id')).toBe(lastRun+':final:answer');
    const text=host.querySelector('.conversation')!.textContent!;
    expect(text.indexOf('使用场景：通勤')).toBeLessThan(text.indexOf('更推荐便携款。'));
    expect(host.querySelector('.product-grid--single')).not.toBeNull();expect(host.querySelector('.product-visual')).toBeNull();
    await act(async()=>root.unmount());root=createRoot(host);await act(async()=>root.render(<App/>));
    await vi.waitFor(()=>expect(host.querySelector('.shopping-form-receipt')).not.toBeNull());
    expect(host.querySelector('.shopping-form')).toBeNull();expect(host.querySelector('.shopping-form-receipt button')).toBeNull();
    expect(posts).toBe(2);expect(writes).toBe(1);
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});
