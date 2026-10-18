import {renderToStaticMarkup} from "react-dom/server";
import {expect, it} from "vitest";
import ProductCards, {money} from "../src/components/ProductCards";
import type {ProductCard} from "../src/types";

it("原版紧凑卡片以商品价为主，到手报价另列，不夹带推荐长文案或选择按钮", () => {
  const product: ProductCard = {
    product_id:"P1001", title:"旅行套装", brand:"test", category:"旅行装备", origin_country:"CN",
    price_major:189, currency:"CNY", highlights:[], score:0, default_sku_id:"P1001-S1", quantity:2,
    skus:[{sku_id:"P1001-S1",spec:"标准",price_major:189,currency:"CNY",stock:50}],
    recommendation_reason:"适合旅行收纳", tradeoffs:["只有一种颜色", "评分为快照，仅供参考", "本单仅为推荐与报价，未下单"],
    rating_summary:{average:4,review_count:327}, rating_is_live:false,
    image_url:"/products/wanderlite.png", image_kind:"illustration", image_alt:"AI 生成示意图，非商品实拍",
    landed_price:{items:[{product_id:"P1001",sku_id:"P1001-S1",title:"旅行套装",quantity:2,
      unit_price_minor:18900,source_unit_price_minor:18900,source_currency:"CNY",currency:"CNY",
      subtotal_minor:37800,freight_minor:4000,tariff_minor:0,total_amount_minor:41800}],
      ship_to:"CN",currency:"CNY",subtotal_minor:37800,freight_minor:4000,tariff_minor:0,total_amount_minor:41800},
  };
  const html = renderToStaticMarkup(<ProductCards mode="recommendation" products={[product]} favoriteIds={new Set()} comparedIds={new Set()}
    onFavorite={()=>{}} onCompare={()=>{}} onDetail={()=>{}}/>).replace(/<!--.*?-->/g, "");
  expect(html).toContain(money(418,"CNY"));
  expect(html).toContain("2件");
  expect(html).toContain(money(189,"CNY"));
  expect(html).not.toContain("运费");expect(html).not.toContain("税费");
  expect(html).toContain("配送至 CN");expect(html).not.toContain("P1001-S1");
  expect(html).not.toContain("适合旅行收纳");
  expect(html).not.toContain("需要权衡");
  expect(html).not.toContain("preferred-pick");
  expect(html).not.toContain("选这款");
  expect(html).toContain("看看细节");
  expect(html).not.toContain("评分为快照");
  expect(html).not.toContain("未下单");
  expect(html).toContain("327");
  expect(html).toContain("样例");
  expect(html).not.toContain("<ul");
  expect(product.tradeoffs).toHaveLength(3);
  expect(product.rating_summary?.review_count).toBe(327);
  expect(html).toContain('alt="AI 生成示意图，非商品实拍"');
  expect(html).toContain("示意图");
  expect(html).toContain("非商品实拍");
  expect(html).toContain("非实物照片");
});

it("没有匹配的规格报价时只显示商品价，不把别的规格到手价当作当前金额",()=>{
  const product:ProductCard={product_id:"P1003",title:"通勤包",brand:"test",category:"旅行装备",origin_country:"CN",
    price_major:129,currency:"CNY",highlights:[],score:0,default_sku_id:"P1003-S1",
    skus:[{sku_id:"P1003-S1",spec:"黑色",price_major:129,currency:"CNY",stock:10}]};
  const html=renderToStaticMarkup(<ProductCards mode="recommendation" products={[product]} favoriteIds={new Set()} comparedIds={new Set()}
    onFavorite={()=>{}} onCompare={()=>{}} onDetail={()=>{}}/>);
  expect(html).toContain(money(129,"CNY"));expect(html).toContain('商品价');
  expect(html).toContain('到手价待目的地与规格确认');expect(html).not.toContain(money(1024,'CNY'));expect(html).not.toContain('P1003-S1');
});
