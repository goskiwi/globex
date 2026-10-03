import {useEffect,useState,type FormEvent} from "react";
import App from "../App";
import Icon from "./Icon";
import {AUTH_KEY,apiBase,readAuth,type AuthSession} from "../lib/auth";

export default function AuthRoot() {
  const [session,setSession]=useState<AuthSession|null>(null),[restoring,setRestoring]=useState(true);
  const [account,setAccount]=useState("kkqq"),[password,setPassword]=useState("");
  const [busy,setBusy]=useState(false),[error,setError]=useState("");
  function logout(){sessionStorage.removeItem(AUTH_KEY);setSession(null);setPassword("");setError("");}
  useEffect(()=>{
    let alive=true;
    async function restore(){
      try {
        const raw=readAuth(JSON.parse(sessionStorage.getItem(AUTH_KEY)??"null"));
        if(!raw){sessionStorage.removeItem(AUTH_KEY);return;}
        const response=await fetch(`${apiBase}/commerce/auth/me`,{headers:{Authorization:`Bearer ${raw.accessToken}`}});
        if(response.status===401){sessionStorage.removeItem(AUTH_KEY);return;}
        if(!response.ok)throw new Error();
        const data=await response.json();
        if(typeof data.buyerId!=="string"||!data.buyerId)throw new Error();
        if(alive){const verified={...raw,buyerId:data.buyerId};sessionStorage.setItem(AUTH_KEY,JSON.stringify(verified));setSession(verified);}
      }catch{if(alive)setError("暂时无法恢复登录，请重新登录或稍后重试。");}
      finally{if(alive)setRestoring(false);}
    }
    void restore();return()=>{alive=false;};
  },[]);
  async function login(event:FormEvent){
    event.preventDefault();setBusy(true);setError("");
    try {
      const response=await fetch(`${apiBase}/commerce/auth/login`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({account,password})});
      if(!response.ok)throw new Error(response.status===401?"账号或密码不正确。":"登录服务暂不可用，请稍后重试。");
      const signed=readAuth(await response.json());
      if(!signed)throw new Error("登录响应无效，请重试。");
      sessionStorage.setItem(AUTH_KEY,JSON.stringify(signed));setPassword("");setSession(signed);
    }catch(err){setError(err instanceof Error?err.message:"登录失败，请重试。");}
    finally{setBusy(false);}
  }
  if(restoring)return <main className="auth-shell"><p role="status">正在恢复登录…</p></main>;
  if(session)return <App key={session.buyerId} session={session} onLogout={logout}/>;
  return <main className="auth-shell"><section className="login-panel">
    <div className="login-brand"><Icon name="globe"/><span>Globex</span></div>
    <h1>继续你的选购</h1><p>选择账号，登录后查看自己的对话与好物。</p>
    <form onSubmit={login}>
      <label htmlFor="login-account">账号</label>
      <div className="login-account-control">
        <select id="login-account" value={account} onChange={e=>setAccount(e.target.value)} disabled={busy}>
          <option value="kkqq">kkqq</option><option value="root">root</option>
        </select>
        <Icon name="chevronDown"/>
      </div>
      <label htmlFor="login-password">密码</label>
      <input id="login-password" type="password" autoComplete="current-password" required value={password}
        onChange={e=>setPassword(e.target.value)} placeholder="请输入密码" disabled={busy}/>
      {error&&<p className="login-error" role="alert">{error}</p>}
      <button type="submit" className="primary" disabled={busy}>{busy?"正在登录…":"登录"}</button>
    </form>
  </section></main>;
}
