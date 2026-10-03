// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import App from './WorkspaceFixture';

it('运行中的对话不锁住历史浏览；切换只断订阅，不取消服务端运行',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  localStorage.clear();sessionStorage.clear();
  vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  let subscribed=false,detached=false,cancelled=0,posts=0;
  const run={runId:'running-r',threadId:'active-s',status:'running',state:{},
    messages:[{id:'current-user',role:'user',content:'合成正在处理的问题'}],
    input:{messages:[{id:'current-user',role:'user',content:'合成正在处理的问题'}]}};
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(init?.method==='POST')posts++;
    if(url.includes('/cancel'))cancelled++;
    if(url.includes('/sessions?'))return Response.json({sessions:[
      {id:'active-s',title:'正在处理的对话',updatedAt:2},{id:'older-s',title:'之前的对话',updatedAt:1}]});
    if(url.includes('/sessions/older-s'))return Response.json({productViews:[],messages:[
      {id:'older-r:final:answer',role:'assistant',content:'已打开旧对话原文'}],
      run:{runId:'older-r',threadId:'older-s',status:'completed',messages:[],state:{}}});
    if(url.includes('/sessions/active-s'))return Response.json({run});
    if(url.includes('/runs/running-r?'))return Response.json(run);
    if(url.includes('/runs/running-r/events'))return new Response(new ReadableStream({start(controller){
      subscribed=true;
      controller.enqueue(new TextEncoder().encode('id: running-r:1\ndata: '+JSON.stringify({type:'RUN_STARTED',runId:'running-r',threadId:'active-s'})+'\n\n'));
      init?.signal?.addEventListener('abort',()=>{detached=true;controller.error(new DOMException('断开订阅','AbortError'));},{once:true});
    }}),{headers:{'Content-Type':'text/event-stream'}});
    return Response.json({sessions:[],confirmations:[],skills:[],form:null,revision:0,products:[]});
  }));
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  try{
    await act(async()=>root.render(<App/>));
    for(let i=0;i<50&&!subscribed;i++)await act(async()=>{await new Promise(r=>setTimeout(r,5));});
    expect(subscribed).toBe(true);
    const nav=[...host.querySelectorAll<HTMLButtonElement>('button')].find(b=>b.textContent==='对话历史')!;
    await act(async()=>nav.click());
    const entries=[...host.querySelectorAll<HTMLButtonElement>('.history-entry')];
    expect(entries).toHaveLength(2);expect(entries.every(b=>!b.disabled)).toBe(true);
    await act(async()=>entries[0].parentElement!.querySelector<HTMLButtonElement>('.session-more')!.click());
    expect(document.querySelector<HTMLButtonElement>('[role=menuitem].danger')?.disabled).toBe(true);
    await act(async()=>document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true})));
    await act(async()=>entries[1].click());
    expect(host.textContent).toContain('已打开旧对话原文');
    expect(detached).toBe(true);expect(cancelled).toBe(0);expect(posts).toBe(0);
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});
