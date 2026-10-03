import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import ShoppingPlans from "../src/components/ShoppingPlans";
import ExecutionProcess from '../src/components/ExecutionProcess';
import EventTimeline from "../src/components/EventTimeline";
import { readPublishedSkills, readSkillUsages } from "../src/lib/skills";
import { CommerceClient } from "./clientFixture";
import type { PublishedSkill } from "../src/types";
const skill: PublishedSkill={id:"weekend-travel",version:"1",title:"周末出游",description:"比较旅行装备",scope:"shopping",content_hash:"a".repeat(64),expires_at:null};
const catalog=(skills:unknown[])=>({capability_digest:"c".repeat(64),skills});
const sse=(events:unknown[])=>new Response(events.map(e=>"data: "+JSON.stringify(e)+"\n\n").join(""),{headers:{"Content-Type":"text/event-stream"}});

it("目录只展示元数据，不提供选择执行动作",()=>{
  const html=renderToStaticMarkup(<ShoppingPlans skills={[skill]} status="ready" error={null} onRefresh={()=>{}} />);
  expect(html).toContain("无需选择方案");
  expect(html).toContain(skill.title);
  expect(html).not.toContain("aria-pressed");
  expect(html).not.toContain("选择方案</");
});
it("目录剔除失效资料，不把错误响应当空库",()=>{
  expect(readPublishedSkills(catalog([skill]))).toEqual([skill]);
  expect(readPublishedSkills(catalog([{...skill,expires_at:"2000-01-01"}]))).toEqual([]);
  expect(()=>readPublishedSkills({skills:[]})).toThrow();
});
it("已读取只来自有效工具结果，未完成与失败不伪装成功",()=>{
  expect(readSkillUsages([{toolCallId:"x",status:"used"}])).toEqual([]);
  const used={toolCallId:"x",status:"used" as const,id:skill.id,version:skill.version,title:skill.title,contentHash:skill.content_hash};
  const html=renderToStaticMarkup(<><ExecutionProcess process={{runId:'r',userMessageId:'u',status:'running',steps:[
    {id:'x',label:'读取选购方案',status:'completed',summary:'已读取：周末出游'},
    {id:'err',label:'读取选购方案',status:'failed',summary:'选购方案未能读取'}]}}/><EventTimeline events={[]} skillUsages={[used]} running={false}/></>);
  expect(html.match(/已读取：周末出游/g)).toHaveLength(1);
  expect(html).toContain("工具成功返回");
  expect(html).not.toContain("选购方案未能读取"); // 详细失败保留在可展开过程，不在折叠状态堆日志。
  expect(renderToStaticMarkup(<EventTimeline events={[]} skillUsages={[{...used,status:'reading'}]} running={false}/>)).toContain('方案读取未完成');
});
it("普通请求和目录读取使用同一买家身份，正文不从客户端发送",async()=>{
  const bodies:any[]=[];
  const client=new CommerceClient({url:"/commerce/ag-ui/run",buyerId:"test-buyer",accessToken:"local-test-token",fetch:async(url,init)=>{
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer local-test-token");
    if(String(url).includes("/skills?"))return Response.json(catalog([skill]));
    const body=JSON.parse(String(init.body));bodies.push(body);
    return sse([{type:"RUN_STARTED",threadId:body.threadId,runId:body.runId},{type:"RUN_FINISHED",threadId:body.threadId,runId:body.runId}]);
  }});
  await client.refreshSkills();await client.submit("周末去旅行");
  expect(bodies[0].forwardedProps).not.toHaveProperty("selectedSkill");
  expect(bodies[0].messages.at(-1).content).toBe("周末去旅行");
  expect(client.getSnapshot().skillUsages).toEqual([]);
});
it("SDK消费实际状态投影，后续快照替换而非沿用旧成功", async () => {
  let calls = 0;
  const client = new CommerceClient({ url: "/commerce/ag-ui/run", fetch: async (_, init) => {
    const body = JSON.parse(String(init.body));calls++;
    return sse([{ type: "RUN_STARTED", threadId: body.threadId, runId: body.runId },
      { type: "STATE_SNAPSHOT", snapshot: { products: [], skillUsages: calls === 1 ? [{ toolCallId: "skill-call", id: skill.id, title: skill.title, version: skill.version, contentHash: skill.content_hash, status: "used" }] : [] } },
      { type: "RUN_FINISHED", threadId: body.threadId, runId: body.runId }]);
  } });
  await client.submit("按方案整理");
  expect(client.getSnapshot().skillUsages[0]).toMatchObject({ title: skill.title, version: skill.version, status: "used" });
  await client.submit("现在直接找耳机");
  expect(client.getSnapshot().skillUsages).toEqual([]);
});

it("503和409不会伪装为真实空库，刷新响应有顺序保护", async () => {
  let finishFirst: ((response: Response) => void) | undefined;
  let calls = 0;
  const client = new CommerceClient({ url: "/commerce/ag-ui/run", fetch: async () => {
    calls++;
    if (calls === 1) return new Promise<Response>((resolve) => { finishFirst = resolve; });
    return Response.json({}, { status: 409 });
  } });
  const first = client.refreshSkills();
  await client.refreshSkills();
  finishFirst!(Response.json(catalog([skill])));await first;
  expect(client.getSnapshot().skillsStatus).toBe("error");
  expect(client.getSnapshot().skillsError).toContain("刚刚更新");
  expect(client.getSnapshot().skills).toEqual([]);
});
