import type {RunProcess,ProcessStatus,ProcessStepStatus} from '../types';

const statuses:ProcessStatus[]=['queued','running','completed','partial','failed','cancelled','interrupted','waiting_input','waiting_confirmation'];
const stepStatuses:ProcessStepStatus[]=['running','completed','partial','failed','cancelled','waiting_input','waiting_confirmation','unconfirmed'];
const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);

/** 只读取本次流程投影，不从旧progress、工具文本或模型正文重建另一套状态。 */
export function readProcess(value:unknown):RunProcess|null {
  if(!record(value)||typeof value.runId!=='string'||typeof value.userMessageId!=='string'||!value.userMessageId
    ||!statuses.includes(value.status as ProcessStatus)||!Array.isArray(value.steps))return null;
  const steps=new Map<string,RunProcess['steps'][number]>();
  for(const item of value.steps){
    if(!record(item)||typeof item.id!=='string'||typeof item.label!=='string'||typeof item.summary!=='string'
      ||!stepStatuses.includes(item.status as ProcessStepStatus))continue;
    steps.set(item.id,{id:item.id,label:item.label,summary:item.summary,status:item.status as ProcessStepStatus});
  }
  return {runId:value.runId,userMessageId:value.userMessageId,status:value.status as ProcessStatus,steps:[...steps.values()]};
}
export function readProcesses(value:unknown):RunProcess[] {
  return Array.isArray(value)?value.flatMap(item=>{const process=readProcess(item);return process?[process]:[];}):[];
}
export function upsertProcess(processes:RunProcess[],process:RunProcess):RunProcess[] {
  return processes.some(item=>item.runId===process.runId)
    ?processes.map(item=>item.runId===process.runId?process:item):[...processes,process];
}
