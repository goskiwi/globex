/** 当前协议的测试数据构造，不接受旧整轮交付状态。 */
export function presented(kind:string,payload:unknown,runId:string):any[] {
  return payload?[{id:`${runId}:${kind}`,runId,kind,status:'completed',payload}]:[];
}
export function shown(kind:string,records:any[]):any[] {
  return records.flatMap(record=>presented(kind,record,record.runId));
}
