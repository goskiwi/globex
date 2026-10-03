// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import {frontendEntry,useFrontendUpdate} from '../src/hooks/useFrontendUpdate';

it('只从实际模块入口识别部署差异，不从商品字段或文本判断更新',()=>{
  expect(frontendEntry('<script type="module" src="/assets/index-next.js"></script>','http://127.0.0.1:5173/'))
    .toBe('http://127.0.0.1:5173/assets/index-next.js');
  expect(frontendEntry('not an application page','http://localhost:5173/')).toBeNull();
});

it('检查服务不可用时保留页面，不产生重载或阻断',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const script=document.createElement('script');script.type='module';script.src='/assets/index-current.js';document.head.append(script);
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host),reload=vi.fn();
  vi.stubGlobal('fetch',vi.fn(async()=>{throw new Error('合成网络失败');}));
  function Test(){const pending=useFrontendUpdate(true,reload);return <span>{pending?'更新':'原页面仍可用'}</span>;}
  try{await act(async()=>root.render(<Test/>));expect(host.textContent).toBe('原页面仍可用');expect(reload).not.toHaveBeenCalled();}
  finally{await act(async()=>root.unmount());script.remove();host.remove();vi.unstubAllGlobals();}
});

it('驻留旧页面发现新版时，先保留输入/运行，空闲后刷新；检查失败不阻断页面',async()=>{
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const script=document.createElement('script');script.type='module';script.src='/assets/index-old.js';document.head.append(script);
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host),reload=vi.fn();
  vi.stubGlobal('fetch',vi.fn(async()=>new Response('<script type="module" src="/assets/index-new.js"></script>')));
  function Test({safe}:{safe:boolean}){const pending=useFrontendUpdate(safe,reload);return <span>{pending?'更新待加载':'当前页面'}</span>;}
  try{
    await act(async()=>root.render(<Test safe={false}/>));
    expect(host.textContent).toBe('更新待加载');expect(reload).not.toHaveBeenCalled();
    await act(async()=>root.render(<Test safe={true}/>));
    expect(reload).toHaveBeenCalledTimes(1);
    await act(async()=>window.dispatchEvent(new Event('focus')));
    expect(reload).toHaveBeenCalledTimes(1);
    const request=(globalThis.fetch as any).mock.calls[0][1];expect(request.cache).toBe('no-store');
  }finally{await act(async()=>root.unmount());script.remove();host.remove();vi.unstubAllGlobals();}
});
