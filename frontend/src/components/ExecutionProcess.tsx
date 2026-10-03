import {useEffect,useState} from 'react';
import type {RunProcess,ConnectionState,ProcessStepStatus} from '../types';
import Icon from './Icon';

const labels={queued:'正在处理你的需求',running:'正在处理你的需求',completed:'本次过程 · 已完成',partial:'本次过程 · 部分完成',
  failed:'本次过程 · 执行失败',cancelled:'本次过程 · 已停止',interrupted:'本次过程 · 执行中断',
  waiting_input:'本次过程 · 等待补充信息',waiting_confirmation:'本次过程 · 等待用户确认'};
const statusLabels:Record<ProcessStepStatus,string>={running:'执行中',completed:'已完成',partial:'部分完成',failed:'失败',
  cancelled:'已停止',unconfirmed:'结果未确认',waiting_input:'需补充信息',waiting_confirmation:'等待确认'};

export default function ExecutionProcess({process,connection='idle',inputResolved=false,stopRequested=false}:{process:RunProcess;connection?:ConnectionState;inputResolved?:boolean;stopRequested?:boolean}){
  const [expanded,setExpanded]=useState(false);
  const running=['queued','running'].includes(process.status);
  const open=expanded;
  useEffect(()=>{if(!running)setExpanded(false);},[running]);
  const current=process.steps.filter(step=>step.status==='running').at(-1)??process.steps.filter(step=>step.status==='completed').at(-1);
  const heading=stopRequested&&running?'正在请求停止，等待服务端确认':running&&connection==='reconnecting'?'连接中断，正在恢复执行进度':running&&connection==='disconnected'
    ?'连接已中断，已有过程已保留':running&&connection==='connecting'?'正在连接选购服务':process.status==='waiting_input'&&inputResolved?'本次过程 · 已补充信息':labels[process.status];
  return <section className="execution-process" aria-label="本次执行过程" data-run-id={process.runId} aria-live="polite">
    <button className="execution-heading" onClick={()=>setExpanded(!expanded)} aria-expanded={open}>
      <span><Icon name={running?'clock':process.status==='completed'?'check':'info'}/>{heading}</span>
      {process.steps.length>0&&<small>{process.steps.length}个步骤</small>}
      <Icon name="chevronDown"/>
    </button>
    {running&&!open&&<div className="execution-current"><span>{current?.label||'正在处理本轮需求'}</span>
      <p>{current?.summary||(connection==='reconnecting'?'正在恢复接收进度，不会重新执行工具。':'可展开查看实际查询和核对结果。')}</p></div>}
    {open&&<ol className="execution-steps">{process.steps.map(step=><li key={step.id} data-step-id={step.id} className={`execution-step execution-step--${step.status}`}>
      <Icon name={step.status==='completed'?'check':step.status==='running'?'clock':step.status==='cancelled'?'stop':'info'}/>
      <div><div className="execution-step-title"><span>{step.label}</span><small>{statusLabels[step.status]}</small></div>
        {step.summary&&<p>{step.summary}</p>}</div>
    </li>)}{running&&!process.steps.some(s=>s.status==='running')&&<li className="execution-wait"><Icon name="clock"/><span>{connection==='disconnected'?'执行状态尚未确认，可恢复连接查看已保存进度。':connection==='reconnecting'?'正在重新接收服务端进度，不会重新执行工具。':'正在继续处理，本轮尚未结束…'}</span></li>}</ol>}
  </section>;
}
