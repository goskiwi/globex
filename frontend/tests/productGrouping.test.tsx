import {renderToStaticMarkup} from 'react-dom/server';
import {expect,it} from 'vitest';
import ProductCards from '../src/components/ProductCards';
import {groupProductRecords,initialOffer} from '../src/lib/productGroups';
import type {ProductCard} from '../src/types';

const card=(id:string,platform:string,canonical='CAN-A'):ProductCard=>({
  product_id:id,canonical_product_id:canonical,title:'同一标题的颈枕',brand:'Roamix',category:'旅行装备',origin_country:'US',
  source_platform:platform,description:'可拆洗纯棉外套，单侧支撑。',weight_kg:0.15,price_major:198,currency:'CNY',
  score:1,highlights:[],default_sku_id:id+'-S1',rating_summary:{average:platform==='amazon'?4.2:4.3,review_count:80},
  skus:[{sku_id:id+'-S1',spec:'石墨黑',price_major:198,currency:'CNY',stock:0},
        {sku_id:id+'-S2',spec:'雾灰',price_major:213.99,currency:'CNY',stock:platform==='ebay'?95:0}],
});

it('同款一张可点击商品卡，不把平台规格表塞进卡片，也不把缺货低价当主价',()=>{
  const products=[card('P3020','amazon'),card('P3021','ebay')];
  const groups=groupProductRecords(products);expect(groups).toHaveLength(1);expect(groups[0].records).toEqual(products);
  const html=renderToStaticMarkup(<ProductCards mode="view" products={products} onDetail={()=>{}}/>);
  expect(html.match(/<article/g)).toHaveLength(1);
  expect(html).not.toContain('product-visual');expect(html).toContain('product-topline');expect(html).toContain('查看详情');expect(html).toContain('选这款');
  expect(html).toContain('Amazon');expect(html).toContain('eBay');expect(html).toContain('213.99');
  expect(html).not.toContain('198.00');expect(html).not.toContain('<details');expect(html).not.toContain('<table');
  expect(html).not.toContain('平台记录');
  expect(html).not.toContain('preferred-pick');expect(html).not.toContain('compare-check');expect(html).not.toContain('准备下单');
});

it('不同同款主键或缺少同款主键不能靠标题合并，资料差异不宣称为公共事实',()=>{
  const a=card('P3020','amazon'),b=card('P3021','ebay','CAN-B');
  expect(groupProductRecords([a,b])).toHaveLength(2);
  expect(groupProductRecords([{...a,canonical_product_id:undefined},{...b,canonical_product_id:undefined}])).toHaveLength(2);
  const group=groupProductRecords([a,{...b,canonical_product_id:'CAN-A',weight_kg:0.17}])[0];
  expect(group.sharedFields).not.toContain('weight_kg');
  expect(group.records.map(p=>p.weight_kg)).toEqual([0.15,0.17]);
});

it('查看缺货商品仍能打开详情，未统一币种时不计算虚构最低价',()=>{
  const unavailable=renderToStaticMarkup(<ProductCards mode="view" products={[card('P3020','amazon')]} onDetail={()=>{}}/>);
  expect(unavailable).toContain('当前缺货');expect(unavailable).toContain('查看详情');
  const a={...card('P3020','amazon'),skus:[{sku_id:'P3020-S1',spec:'黑色',price_major:198,currency:'CNY',stock:1}]};
  const b={...card('P3021','ebay'),skus:[{sku_id:'P3021-S1',spec:'黑色',price_major:30,currency:'USD',stock:1}]};
  const html=renderToStaticMarkup(<ProductCards mode="view" products={[a,b]} onDetail={()=>{}}/>);
  expect(html).toContain('查看各平台报价');expect(html).not.toContain('¥30');
});

it('单平台查看也优先有货报价，只有推荐明确交付的规格才锁定初始选择',()=>{
  const product=card('P3021','ebay');
  const group=groupProductRecords([product])[0];
  expect(initialOffer(group)?.sku.sku_id).toBe('P3021-S2');
  expect(initialOffer({...group,initialSkuId:'P3021-S1'})?.sku.sku_id).toBe('P3021-S1');
});

it('同款起价和初始规格优先当前可配送的有货记录，仍保留全部平台详情',()=>{
  const deliverable={...card('P3021','ebay'),skus:[{sku_id:'P3021-S2',spec:'灰色',price_major:213.99,currency:'CNY',stock:95,constraint_issues:['over_price_cap']}]};
  const other={...card('P3022','etsy'),skus:[{sku_id:'P3022-S1',spec:'黑色',price_major:212,currency:'CNY',stock:100,constraint_issues:['ship_to_unavailable','over_price_cap']}]};
  const group=groupProductRecords([other,deliverable])[0];
  expect(group.records).toHaveLength(2);expect(initialOffer(group)?.product.product_id).toBe('P3021');
  const html=renderToStaticMarkup(<ProductCards mode="view" products={[other,deliverable]} onDetail={()=>{}}/>);
  expect(html).toContain('213.99');expect(html).not.toContain('212.00');
});
