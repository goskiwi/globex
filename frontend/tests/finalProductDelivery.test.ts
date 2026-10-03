import {expect,it} from "vitest";
import {CommerceClient} from "./clientFixture";

it("整轮部分完成仍展示服务端已确认的成功交付，不把研究停止当作卡片失败",async()=>{
  const product={product_id:"P1003",title:"背包",price_major:129,currency:"CNY",brand:"test",category:"旅行装备",
    origin_country:"CN",score:1,highlights:[],default_sku_id:"P1003-S1",
    skus:[{sku_id:"P1003-S1",spec:"黑",price_major:129,currency:"CNY",stock:10}]};
  const client=new CommerceClient({url:"/commerce/ag-ui/run",fetch:async(_,init)=>{
    const body=JSON.parse(String(init.body));
    const events=[{type:"RUN_STARTED",threadId:body.threadId,runId:body.runId},
      {type:"MESSAGES_SNAPSHOT",messages:[...body.messages,{id:"answer",role:"assistant",content:"已确认这款，其他候选尚未确认。"}]},
      {type:"STATE_SNAPSHOT",snapshot:{status:"partial",executionStatus:"partial",stopReason:"model_call_limit",
        productDeliveryComplete:true,deliveredRunId:body.runId,comparison:null,recommendation:{mode:"alternatives",
          guidance:"已确认这款，其他候选尚未确认。",preferred_sku_id:"P1003-S1",dimensions:["用途"],max_items:12,
          hits:[product],quote:null,result_ref:"ctx_delivered",unverified_requirements:[]}}},
      {type:"RUN_FINISHED",threadId:body.threadId,runId:body.runId}];
    return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(""),{headers:{"Content-Type":"text/event-stream"}});
  }});
  await client.submit("研究背包");
  expect(client.getSnapshot().recommendation?.hits[0].product_id).toBe("P1003");
  expect(client.getSnapshot().processes.at(-1)?.status).toBe('partial');
  expect(client.getSnapshot().error).toBeNull();
});

it("旧候选字段和连续检索快照不能进入商品区或历史",async()=>{
  const client=new CommerceClient({url:"/commerce/ag-ui/run",fetch:async(_,init)=>{
    const body=JSON.parse(String(init.body));
    const events=[{type:"RUN_STARTED",threadId:body.threadId,runId:body.runId},
      ...["A","B","C"].map(id=>({type:"STATE_SNAPSHOT",snapshot:{
        products:[{product_id:id,title:id,price_major:1,currency:"CNY",skus:[]}],
        searchCompleted:true,progress:[]}})),
      {type:"MESSAGES_SNAPSHOT",messages:[...body.messages,{id:"final",role:"assistant",content:"暂未形成推荐"}]},
      {type:"RUN_FINISHED",threadId:body.threadId,runId:body.runId}];
    return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(""),{headers:{"Content-Type":"text/event-stream"}});
  }});
  const seen:unknown[]=[];client.subscribe(()=>seen.push(client.getSnapshot()));
  await client.submit("找背包");
  for(const value of seen){
    expect(value).not.toHaveProperty("products");
    expect(value).not.toHaveProperty("searchCompleted");
    expect(value).toMatchObject({recommendation:null,comparison:null,productHistory:[]});
  }
});

it("追问和空快照保留交付，新的交付才替换并归档一次",async()=>{
  const product={product_id:"P1003",title:"背包",price_major:129,currency:"CNY",skus:[{sku_id:"P1003-S1",spec:"黑",price_major:129,currency:"CNY",stock:10}],
    default_sku_id:"P1003-S1",brand:"test",category:"旅行装备",origin_country:"CN",score:1,highlights:[]};
  const replacement={...product,product_id:"P1049",default_sku_id:"P1049-S1",skus:[{sku_id:"P1049-S1",spec:"黑",price_major:129,currency:"CNY",stock:10}],title:"另一款背包"};
  let count=0,firstRun="";
  const client=new CommerceClient({url:"/commerce/ag-ui/run",fetch:async(_,init)=>{
    const body=JSON.parse(String(init.body));count++;
    if(count===1)firstRun=body.runId;
    const recommendation=count===2?null:{mode:"alternatives",guidance:"更在意轻便时选择这款。",preferred_sku_id:null,dimensions:["用途"],max_items:12,hits:[count===1?product:replacement],quote:null,
      result_ref:`ctx_${count}`,unverified_requirements:[]};
    const events=[{type:"RUN_STARTED",threadId:body.threadId,runId:body.runId},
      {type:"STATE_SNAPSHOT",snapshot:{recommendation,comparison:null,deliveredRunId:recommendation?body.runId:null}},
      {type:"RUN_FINISHED",threadId:body.threadId,runId:body.runId}];
    return new Response(events.map(e=>`data: ${JSON.stringify(e)}\n\n`).join(""),{headers:{"Content-Type":"text/event-stream"}});
  }});
  await client.submit("推荐背包");
  const seen:ReturnType<typeof client.getSnapshot>[]=[];
  const unsubscribe=client.subscribe(()=>seen.push(client.getSnapshot()));
  await client.submit("核对这个背包库存");unsubscribe();
  expect(seen.every(s=>s.recommendation?.hits[0].product_id==="P1003")).toBe(true);
  expect(client.getSnapshot().deliveredRunId).toBe(firstRun);
  expect(client.getSnapshot().productHistory).toEqual([]);
  await client.submit("换一款推荐");
  expect(client.getSnapshot().recommendation?.hits[0].product_id).toBe("P1049");
  expect(client.getSnapshot().productHistory).toEqual([expect.objectContaining({runId:firstRun,products:[product]})]);
  client.reset();
  expect(client.getSnapshot().recommendation).toBeNull();
});
