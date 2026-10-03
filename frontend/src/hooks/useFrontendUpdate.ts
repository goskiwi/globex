import {useEffect,useRef,useState} from "react";

/** 复用Vite现有入口文件名识别部署变更，不另造版本、hash或业务结果兜底。 */
export function frontendEntry(html:string,base:string){
  const src=new DOMParser().parseFromString(html,'text/html')
    .querySelector<HTMLScriptElement>('script[type="module"][src]')?.getAttribute('src');
  return src?new URL(src,base).href:null;
}

export function useFrontendUpdate(canReload:boolean,reload:()=>void=()=>window.location.reload()){
  const [available,setAvailable]=useState(false);
  const safe=useRef(canReload),action=useRef(reload);
  const pending=useRef(false),requested=useRef(false);
  safe.current=canReload;action.current=reload;
  const refreshIfSafe=()=>{
    if(pending.current&&safe.current&&!requested.current&&!document.querySelector('[role="dialog"]')){
      requested.current=true;action.current();
    }
  };
  useEffect(()=>{
    const current=document.querySelector<HTMLScriptElement>('script[type="module"][src]')?.src;
    // 开发服务器由HMR接管；实际打包页面才需要部署衔接。
    if(!current||!new URL(current).pathname.startsWith('/assets/'))return;
    let disposed=false,checking=false;
    const check=async()=>{
      if(disposed||checking||document.visibilityState==='hidden')return;
      if(pending.current){refreshIfSafe();return;}
      checking=true;
      try{
        const response=await fetch(new URL('/',window.location.href),{cache:'no-store',headers:{Accept:'text/html'}});
        if(!response.ok)return;
        const next=frontendEntry(await response.text(),window.location.href);
        if(!disposed&&next&&next!==current){pending.current=true;setAvailable(true);}
      }catch{/* 版本检查失败不阻断购物和历史浏览。 */}
      finally{checking=false;}
    };
    window.addEventListener('focus',check);document.addEventListener('visibilitychange',check);
    void check();
    return()=>{disposed=true;window.removeEventListener('focus',check);document.removeEventListener('visibilitychange',check);};
  },[]);
  useEffect(()=>{
    if(available)refreshIfSafe();
  },[available,canReload]);
  return available;
}
