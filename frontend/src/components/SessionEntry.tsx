import {useEffect,useRef,useState} from "react";
import {createPortal} from "react-dom";
import type {SessionSummary} from "../types";
import Icon from "./Icon";
import Modal from "./Modal";

export default function SessionEntry({item,compact=false,current=false,deleteDisabled,onOpen,onRename,onDelete}:{
  item:SessionSummary;compact?:boolean;current?:boolean;deleteDisabled:boolean;
  onOpen:(id:string)=>void;onRename:(id:string,title:string)=>Promise<void>;onDelete:(id:string)=>Promise<void>;
}){
  const [menu,setMenu]=useState<{left:number;top:number}|null>(null),[action,setAction]=useState<"rename"|"delete"|null>(null);
  const [title,setTitle]=useState(item.title),[busy,setBusy]=useState(false),[error,setError]=useState("");
  const trigger=useRef<HTMLButtonElement>(null),popup=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    if(!menu)return;
    popup.current?.querySelector<HTMLButtonElement>('button')?.focus();
    const outside=(e:PointerEvent)=>{if(!popup.current?.contains(e.target as Node)&&!trigger.current?.contains(e.target as Node))setMenu(null);};
    const close=()=>setMenu(null);
    const key=(e:KeyboardEvent)=>{if(e.key==="Escape"){setMenu(null);trigger.current?.focus();}};
    document.addEventListener('pointerdown',outside);document.addEventListener('keydown',key);
    window.addEventListener('scroll',close,true);window.addEventListener('resize',close);
    return()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',key);
      window.removeEventListener('scroll',close,true);window.removeEventListener('resize',close);};
  },[menu]);
  function choose(next:"rename"|"delete"){setMenu(null);setAction(next);setTitle(item.title);setError("");}
  async function submit(){
    setBusy(true);setError("");
    try{if(action==="rename")await onRename(item.id,title);else await onDelete(item.id);setAction(null);}
    catch(err){setError(err instanceof Error?err.message:"操作失败，请重试。");}
    finally{setBusy(false);}
  }
  return <div className={`session-row ${compact?'compact':''}`}>
    <button className={compact?'history-short':'history-entry'} onClick={()=>onOpen(item.id)}
      title={item.title} aria-current={current?'page':undefined}>
      {compact?item.title:<><span className="history-icon"><Icon name="chat"/></span><span><strong>{item.title}</strong>
        <small>{new Date(item.updatedAt).toLocaleString('zh-CN')}{current?' · 当前选购':''}</small></span><Icon name="arrow"/></>}
    </button>
    <button ref={trigger} className="session-more" aria-label={`管理对话：${item.title}`}
      aria-haspopup="menu" aria-expanded={!!menu} onClick={()=>{
        if(menu){setMenu(null);return;}const rect=trigger.current!.getBoundingClientRect();
        setMenu({left:Math.max(8,Math.min(rect.right-160,window.innerWidth-168)),top:Math.max(8,Math.min(rect.bottom+4,window.innerHeight-104))});
      }}>⋯</button>
    {menu&&createPortal(<div ref={popup} role="menu" className="session-menu" style={menu} aria-label="对话操作">
      <button role="menuitem" onClick={()=>choose('rename')}>重命名</button>
      <button role="menuitem" className="danger" disabled={deleteDisabled} onClick={()=>choose('delete')}>删除</button>
    </div>,document.body)}
    {action&&<Modal title={action==='rename'?'重命名对话':'删除对话？'} onClose={()=>{if(!busy)setAction(null);}}>
      <form className="session-action-form" onSubmit={e=>{e.preventDefault();void submit();}}>
        {action==='rename'?<><label htmlFor={`session-title-${item.id}`}>标题</label>
          <input id={`session-title-${item.id}`} value={title} onChange={e=>setTitle(e.target.value)} required maxLength={100} disabled={busy}/></>
          :<p>删除“{item.title}”及其对话记录。订单、收藏和长期偏好保留。</p>}
        {error&&<p role="alert">{error}</p>}
        <div><button type="button" disabled={busy} onClick={()=>setAction(null)}>取消</button>
          <button type="submit" className={action==='delete'?'danger':'primary'} disabled={busy||action==='delete'&&deleteDisabled||action==='rename'&&!title.trim()}>
            {busy?'处理中…':action==='rename'?'保存':'删除'}</button></div>
      </form>
    </Modal>}
  </div>;
}
