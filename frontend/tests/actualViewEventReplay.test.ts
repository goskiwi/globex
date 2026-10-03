import {readFileSync} from 'node:fs';
import {expect,it} from 'vitest';
import {CommerceClient} from './clientFixture';

it('用户故障运行的脱敏事件原样回放：本次确有两张详情且匹配最终回复',async()=>{
  const events=JSON.parse(readFileSync(new URL('./fixtures/view-event-fixture.json',import.meta.url),'utf8'));
  const runId=events[0].runId;
  const client=new CommerceClient({url:'/commerce/ag-ui/run',fetch:async()=>new Response(
    events.map((event:any)=>`data: ${JSON.stringify(event)}\n\n`).join(''),{headers:{'Content-Type':'text/event-stream'}})});
  await client.submitForm('合成请求','form-run-00000000000000000000000000000000');
  const state=client.getSnapshot();
  expect(state.productViews).toHaveLength(1);
  expect(state.productViews[0].runId).toBe(runId);
  expect(state.productViews[0].hits).toHaveLength(2);
  expect(state.messages.some(m=>m.id===`${runId}:final:answer`)).toBe(true);
});
