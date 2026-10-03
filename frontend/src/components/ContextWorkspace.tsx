import { useCallback, useEffect, useRef, useState } from 'react';
import type { WorkspaceRequest } from './BuyerWorkspace';
import './contextWorkspace.css';

type Options={sessionId:string;busy:boolean;pending:boolean;hasMessages:boolean;request:WorkspaceRequest};
type View={revision:number;strategy?:string;summary?:string;working_notice?:string;working?:{goal:string;latest_request:string;filters:Record<string,string|number|string[]|null>;preferences:string[];sort:string|null;selections:Record<string,{product_id:string;sku_id:string;quantity:number}>;comparisons:string[];unverified_requirements:string[]};statistics?:{status?:string;before_tokens?:number;after_tokens?:number}};
// 在 App 会话层调用，切到其它页面时仍继续恢复和轮询整理操作。
export function useContextWorkspace({sessionId,busy,pending,hasMessages,request}:Options){
 const [view,setView]=useState<View|null>(null),[operation,setOperation]=useState(''),[checking,setChecking]=useState(true),[submitting,setSubmitting]=useState(false);
 const [notice,setNotice]=useState(''),[error,setError]=useState('');
 const generation=useRef(0),readRevision=useRef(0),submitLock=useRef(false);
 const refresh=useCallback(async(current:number)=>{
  const revision=++readRevision.current;
  try{
   const data=await request('/context?session_id='+encodeURIComponent(sessionId));
   if(current!==generation.current||revision!==readRevision.current)return;
   setView(data as View);
   const active=data.operation as {operation_id?:string;status?:string}|undefined;
   setOperation(active?.status==='running'&&active.operation_id?active.operation_id:'');
  }catch(e){
   if(current===generation.current&&revision===readRevision.current&&!(e instanceof Error&&'status' in e&&e.status===404))
    setError(e instanceof Error?e.message:'上下文暂时无法读取');
  }finally{if(current===generation.current&&revision===readRevision.current)setChecking(false);}
 },[request,sessionId]);
 useEffect(()=>{
  const current=++generation.current;
  setView(null);setError('');setNotice('');setOperation('');setSubmitting(false);setChecking(true);submitLock.current=false;
  // 没有浏览器消息缓存时也查询数据库，避免漏掉正在执行的整理。
  void refresh(current);
  return()=>{++generation.current;};
 },[refresh]);
 const previous=useRef({busy,hasMessages});
 useEffect(()=>{
  const before=previous.current;previous.current={busy,hasMessages};
  if(!busy&&hasMessages&&(before.busy||!before.hasMessages))void refresh(generation.current);
 },[busy,hasMessages,refresh]);
 useEffect(()=>{
  if(!operation)return;
  const current=generation.current;
  let stopped=false,timer:ReturnType<typeof setTimeout>;
  const poll=async()=>{
   try{
    const data=await request('/context/operations/'+encodeURIComponent(operation));
    if(stopped||current!==generation.current)return;
    if(data.status!=='running'){
     setOperation('');setError('');setNotice(typeof data.message==='string'?data.message:'整理已结束');
     await refresh(current);return;
    }
   }catch(e){if(!stopped)setError(e instanceof Error?e.message:'读取整理进度失败，正在重试');}
   if(!stopped)timer=setTimeout(poll,1500);
  };void poll();return()=>{stopped=true;clearTimeout(timer);};
 },[operation,request,refresh]);
 const running=checking||submitting||!!operation;
 const compact=async()=>{
  if(!view||busy||pending||running||submitLock.current)return;
  const current=generation.current;
  submitLock.current=true;++readRevision.current;setSubmitting(true);setError('');setNotice('');
  try{
   const data=await request('/context/compact','POST',{session_id:sessionId,request_id:crypto.randomUUID(),expected_revision:view.revision});
   if(current!==generation.current)return;
   if(data.status==='running')setOperation(String(data.operation_id));
   else{setNotice(String(data.message??'整理已结束'));await refresh(current);}
  }catch(e){
   if(current===generation.current){setError(e instanceof Error?e.message:'未能整理，原记录保留');await refresh(current);}
  }finally{if(current===generation.current){submitLock.current=false;setSubmitting(false);}}
 };
 return {view,running,notice,error,compact,busy,pending,hasMessages};
}

export default function ContextWorkspace({state}:{state:ReturnType<typeof useContextWorkspace>}){
 const {view,running,notice,error,compact,busy,pending,hasMessages}=state;
 if(!hasMessages)return null;
 return <details className="context-workspace" aria-label="上下文详情">
  <summary>上下文详情</summary>
  <div className="context-workspace-body">
   {view?.working_notice&&<small>{view.working_notice}</small>}
   {view?.working?.goal&&<p>任务需求：{view.working.goal}</p>}
   {view?.working?.latest_request&&<p>最近请求：{view.working.latest_request}</p>}
   {view?.working&&<>
    <ul>{Object.entries(view.working.filters).filter(([,v])=>v!=null&&String(v)!=='').map(([key,v])=><li key={key}>{{price_max_major:'单件商品价上限',landed_budget_major:'到手总预算',target_currency:'币种',ship_to:'配送地',excluded_material_tags:'排除材质',required_material_tags:'必要材质'}[key]??key}：{String(v)}</li>)}</ul>
    {!!view.working.preferences.length&&<p>偏好：{view.working.preferences.join('、')}</p>}
    {view.working.sort&&<p>排序意图（待核验）：{view.working.sort}</p>}
    {Object.values(view.working.selections).map(c=><p key={c.sku_id}>已选商品：{c.sku_id} × {c.quantity}</p>)}
    {!!view.working.comparisons.length&&<p>比较商品：{view.working.comparisons.join('、')}</p>}
    {!!view.working.unverified_requirements.length&&<p>待核验：{view.working.unverified_requirements.join('；')}</p>}
   </>}
   {!view?.working?.goal&&<p>尚未记录任务需求。你可以继续补充，完整对话会保留。</p>}
   <small>这是本次选购的工作记录。价格与库存以重新查询为准；需要修改需求，直接告诉 Globex。</small>
  {view?.summary&&<><p>历史对话摘要：</p><p className="context-summary">{view.summary}</p></>}
  <button type="button" onClick={()=>void compact()} disabled={!view||busy||pending||running}>{running?'正在整理…':'整理上下文'}</button>
  {pending&&<small>请先完成或拒绝待确认操作。</small>}
  {notice&&<p role="status">{notice}</p>}{error&&<p role="alert">{error}</p>}
  </div>
 </details>;
}
