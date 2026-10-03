import {renderToStaticMarkup} from "react-dom/server";
import {expect, it} from "vitest";
import ProductComparison from "../src/components/ProductComparison";
import {money} from "../src/components/ProductCards";
import {comparisonKey, comparisonDifference, upsertComparison} from "../src/lib/comparison";
import {readComparison, CommerceClient} from "./clientFixture";
import type {ProductCard} from "../src/types";

function card(sku: string, unit=18900, quantity=2, ship_to="CN", currency="CNY"): ProductCard {
  const subtotal=unit*quantity, freight=4000, total=subtotal+freight;
  return {product_id:"P1001", title:"旅行套装", brand:"test",category:"旅行装备",origin_country:"CN",
    price_major:unit/100,currency,score:0,highlights:[],default_sku_id:sku,quantity,
    skus:[{sku_id:sku,spec:sku.endsWith("S1")?"军绿色":"沙漠黄",price_major:unit/100,currency,stock:10}],
    recommendation_reason:"颜色偏好",tradeoffs:["颜色不同"],constraint_issues:[],
    landed_price:{ship_to,currency,subtotal_minor:subtotal,freight_minor:freight,tariff_minor:0,total_amount_minor:total,
      items:[{product_id:"P1001",sku_id:sku,title:"旅行套装",quantity,unit_price_minor:unit,currency,
        source_unit_price_minor:unit,source_currency:currency,subtotal_minor:subtotal,freight_minor:freight,tariff_minor:0,total_amount_minor:total}]}};
}

it("同商品不同 SKU 共存，更新其中一个不覆盖另一个",()=>{
  const first=card("P1001-S1"), second=card("P1001-S2",19900);
  const both=upsertComparison([first],second);
  expect(both).toHaveLength(2);
  expect(comparisonKey(first)).not.toBe(comparisonKey(second));
  expect(upsertComparison(both,{...first,quantity:3})).toEqual([{...first,quantity:3},second]);
});

it("同款跨平台比较标明平台和所选规格库存，不向用户展示内部SKU和错误码",()=>{
  const a={...card('P3020-S2'),product_id:'P3020',source_platform:'amazon',landed_price:undefined,constraint_issues:['out_of_stock'],
    skus:[{sku_id:'P3020-S2',spec:'雾灰',price_major:207,currency:'CNY',stock:0}]};
  const b={...card('P3021-S2'),product_id:'P3021',source_platform:'ebay',landed_price:undefined,
    skus:[{sku_id:'P3021-S2',spec:'雾灰',price_major:213.99,currency:'CNY',stock:95}]};
  const html=renderToStaticMarkup(<ProductComparison products={[a,b]}/>);
  expect(html).toContain('Amazon');expect(html).toContain('eBay');expect(html).toContain('缺货');expect(html).toContain('有货');
  expect(html).not.toContain('P3020-S2');expect(html).not.toContain('out_of_stock');
});

it("比较表展示完整费用、SKU 和理由，差价来自结构化报价",()=>{
  const products=[card("P1001-S1"),card("P1001-S2",19900)];
  const html=renderToStaticMarkup(<ProductComparison products={products} preferredSkuId="P1001-S1"/>).replace(/<!--.*?-->/g,"");
  expect(html).toContain("军绿色");expect(html).toContain("沙漠黄");
  expect(html).toContain(money(418,"CNY"));expect(html).toContain(money(438,"CNY"));
  expect(html).toContain(`到手总价相差 ${money(20,"CNY")}`);
  expect(html).toContain("更推荐这款");expect(html).toContain("颜色不同");
});

it.each([
  card("P1001-S2",19900,1),card("P1001-S2",19900,2,"JP"),card("P1001-S2",19900,2,"CN","USD"),
  {...card("P1001-S2"),landed_price:undefined}, {...card("P1001-S2"),default_sku_id:"P1001-S3"},
])("数量、目的地、币种不同或报价缺失，不制造价差",other=>{
  expect(comparisonDifference([card("P1001-S1"),other])).toBeNull();
});

it("未知目的地仅展示商品金额，未报价不显示为零",()=>{
  const products=[{...card("P1001-S1"),landed_price:undefined},{...card("P1001-S2"),landed_price:undefined}];
  const html=renderToStaticMarkup(<ProductComparison products={products}/>);
  expect(html).toContain("商品价");expect(html).toContain(money(189,'CNY'));
  expect(html).not.toContain('运费');expect(html).not.toContain('税费');expect(html).not.toContain('到手价');
  expect(html).not.toContain('不计算价差');
  expect(html).not.toContain("更推荐这款");
});

it("比较结果不接受重复 SKU 或范围外的推荐倾向",()=>{
  const result={guidance:"按颜色选择即可。",hits:[card("P1001-S1"),card("P1001-S2")],preferred_sku_id:null,dimensions:["颜色"],max_items:12,result_ref:"ctx_test",unverified_requirements:[]};
  expect(readComparison(result)?.hits).toHaveLength(2);
  expect(readComparison({...result,hits:[result.hits[0],result.hits[0]]})).toBeNull();
  expect(readComparison({...result,preferred_sku_id:"P9999-S1"})).toBeNull();
});

it("真实 AG-UI 快照消费、本机恢复和追加问题保留比较，新建会话清空",async()=>{
  const data=new Map<string,string>();
  const storage={getItem:(k:string)=>data.get(k)??null,setItem:(k:string,v:string)=>{data.set(k,v);}};
  const comparison={guidance:"按颜色选择即可。",hits:[card("P1001-S1"),card("P1001-S2",19900)],preferred_sku_id:null,dimensions:["颜色"],max_items:12,result_ref:"ctx_test",unverified_requirements:[]};
  let calls=0;
  const fetch=async(_url:any,init:any)=>{
    const body=JSON.parse(String(init.body));
    const state={products:[],comparison:++calls===1?comparison:null,recommendation:null,progress:[]};
    return new Response([
      {type:"RUN_STARTED",threadId:body.threadId,runId:body.runId},
      {type:"STATE_SNAPSHOT",snapshot:state},
      {type:"RUN_FINISHED",threadId:body.threadId,runId:body.runId},
    ].map(e=>`data: ${JSON.stringify(e)}\n\n`).join(""),{headers:{"Content-Type":"text/event-stream"}});
  };
  const first=new CommerceClient({url:"/run",storage,fetch});await first.submit("比较这两种规格");
  expect(first.getSnapshot().comparison?.hits).toHaveLength(2);
  const restored=new CommerceClient({url:"/run",storage,fetch});
  expect(restored.getSnapshot().comparison).toEqual(first.getSnapshot().comparison);
  await restored.submit("核对第一款的库存");
  expect(restored.getSnapshot().comparison).toEqual(first.getSnapshot().comparison);
  restored.reset();expect(restored.getSnapshot().comparison).toBeNull();
});
