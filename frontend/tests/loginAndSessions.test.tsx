// @vitest-environment jsdom
import {act} from "react";
import {createRoot} from "react-dom/client";
import {expect,it,vi} from "vitest";
import AuthRoot from "../src/components/AuthRoot";
import SessionEntry from "../src/components/SessionEntry";
import {CommerceClient} from "../src/lib/commerceClient";
import {AUTH_KEY} from "../src/lib/auth";

async function input(element:HTMLInputElement|HTMLSelectElement,value:string){
  await act(async()=>{Object.getOwnPropertyDescriptor(element instanceof HTMLSelectElement?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value')!.set!.call(element,value);
    element.dispatchEvent(new Event(element instanceof HTMLSelectElement?'change':'input',{bubbles:true}));});
}

it("账号下拉登录、错误密码、退出及重新选择账号；不保存密码",async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;sessionStorage.clear();localStorage.clear();
  vi.stubGlobal('scrollTo',vi.fn());Element.prototype.scrollIntoView=vi.fn();
  const submitted:any[]=[];
  vi.stubGlobal('fetch',vi.fn(async(url:string,init?:RequestInit)=>{
    if(url.endsWith('/auth/login')){
      const data=JSON.parse(String(init?.body));submitted.push(data);
      if(data.password!=='123')return Response.json({detail:'错误'},{status:401});
      return Response.json({buyerId:data.account,accessToken:`signed-${data.account}`,expiresAt:Date.now()+3600000});
    }
    return Response.json({sessions:[],skills:[],confirmations:[],products:[],preferences:[],form:null,revision:0});
  }));
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  try{
    await act(async()=>root.render(<AuthRoot/>));
    expect([...host.querySelectorAll('option')].map(x=>x.value)).toEqual(['kkqq','root']);
    await input(host.querySelector('#login-password')!,'wrong');
    await act(async()=>host.querySelector<HTMLButtonElement>('[type=submit]')!.click());
    expect(host.textContent).toContain('账号或密码不正确');
    await input(host.querySelector('#login-password')!,'123');
    await act(async()=>host.querySelector<HTMLButtonElement>('[type=submit]')!.click());
    expect(host.querySelector('.profile-name')?.textContent).toBe('kkqq');
    expect(sessionStorage.getItem(AUTH_KEY)).not.toContain('password');
    await act(async()=>host.querySelector<HTMLButtonElement>('.logout-button')!.click());
    expect(sessionStorage.getItem(AUTH_KEY)).toBeNull();
    await input(host.querySelector('#login-account')!,'root');await input(host.querySelector('#login-password')!,'123');
    await act(async()=>host.querySelector<HTMLButtonElement>('[type=submit]')!.click());
    expect(host.querySelector('.profile-name')?.textContent).toBe('root');
    expect(submitted.at(-1).account).toBe('root');
  }finally{await act(async()=>root.unmount());host.remove();vi.unstubAllGlobals();}
});

it("三点菜单不打开对话，Portal菜单提供改名与确认删除",async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const open=vi.fn(),rename=vi.fn(async()=>{}),remove=vi.fn(async()=>{});
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host);
  try{
    await act(async()=>root.render(<SessionEntry item={{id:'s',title:'旧标题',updatedAt:1,source:'server'}} compact deleteDisabled={false}
      onOpen={open} onRename={rename} onDelete={remove}/>));
    await act(async()=>host.querySelector<HTMLButtonElement>('.session-more')!.click());
    expect(open).not.toHaveBeenCalled();
    expect(document.querySelector('[role=menu]')?.parentElement).toBe(document.body);
    await act(async()=>document.querySelector<HTMLButtonElement>('[role=menuitem]')!.click());
    await input(document.querySelector('input')!,'新标题');
    await act(async()=>document.querySelector<HTMLButtonElement>('[type=submit]')!.click());
    expect(rename).toHaveBeenCalledWith('s','新标题');
    await act(async()=>host.querySelector<HTMLButtonElement>('.session-more')!.click());
    await act(async()=>document.querySelector<HTMLButtonElement>('[role=menuitem].danger')!.click());
    expect(remove).not.toHaveBeenCalled();
    await act(async()=>document.querySelector<HTMLButtonElement>('[type=submit]')!.click());
    expect(remove).toHaveBeenCalledWith('s');
  }finally{await act(async()=>root.unmount());host.remove();}
});

it("当前对话删除后不再保存旧缓存，账号缓存隔离且拒绝无身份客户端",async()=>{
  const values=new Map<string,string>();
  const storage={getItem:(key:string)=>values.get(key)??null,setItem:(key:string,value:string)=>{values.set(key,value);}};
  const history={id:'s',title:'初始标题',updatedAt:1};let deleted=false,title=history.title;
  const transport=async(url:RequestInfo|URL,init?:RequestInit)=>{
    const path=String(url);
    if(init?.method==='DELETE'){deleted=true;return Response.json({deleted:true});}
    if(init?.method==='PATCH'){title=JSON.parse(String(init.body)).title;return Response.json({title});}
    if(path.includes('/sessions?'))return Response.json({sessions:deleted?[]:[{...history,title}]});
    return Response.json({run:{runId:'r',threadId:'s',status:'completed',messages:[{id:'m',role:'user',content:'原始问题'}],state:{}}});
  };
  expect(()=>new CommerceClient({url:'/run',buyerId:'',accessToken:''})).toThrow('需要登录身份');
  const a=new CommerceClient({url:'/commerce/ag-ui/run',buyerId:'kkqq',accessToken:'signed-a',storage,fetch:transport});
  await a.initialize();await a.renameSession('s','自定义标题');a.detach();
  expect(a.getSnapshot().history[0].title).toBe('自定义标题');
  const staleValues=new Map(values);
  const b=new CommerceClient({url:'/run',buyerId:'root',accessToken:'signed-b',storage});
  expect(b.getSnapshot().messages).toEqual([]);
  await a.deleteSession('s');a.detach();
  expect(a.getSnapshot().sessionId).not.toBe('s');
  const restored=new CommerceClient({url:'/commerce/ag-ui/run',buyerId:'kkqq',accessToken:'signed-a',storage,fetch:transport});
  await restored.initialize();expect(restored.getSnapshot().history).toEqual([]);
  expect([...values.values()].some(value=>value.includes('原始问题'))).toBe(false);
  const staleStorage={getItem:(key:string)=>staleValues.get(key)??null,
    setItem:(key:string,value:string)=>{staleValues.set(key,value);}};
  const otherBrowser=new CommerceClient({url:'/commerce/ag-ui/run',buyerId:'kkqq',accessToken:'signed-a',
    storage:staleStorage,fetch:transport});
  await otherBrowser.initialize();otherBrowser.detach();
  expect(otherBrowser.getSnapshot().history).toEqual([]);
  expect([...staleValues.values()].some(value=>value.includes('原始问题'))).toBe(false);
});
