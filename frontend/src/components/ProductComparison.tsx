import type { ProductCard } from "../types";
import { money, ProductImage } from "./ProductCards";
import {comparisonKey, comparisonQuote, comparisonDifference} from "../lib/comparison";
import {platformLabel} from "../lib/productGroups";

const issueLabels: Record<string, string> = {
  over_price_cap: "商品价超过上限", over_landed_budget: "到手价超过预算",
  product_excluded: "已排除该商品", sku_excluded: "已排除该规格",
  material_excluded: "包含排除材质", material_required_missing: "缺少要求的材质",
  ship_to_unavailable: "不支持当前配送地", category_mismatch: "品类不符",
  out_of_stock: "缺货",
};

/** 同一张表同时用于 Agent 交付与手动快照比较。金额只读结构化数据。 */
export default function ProductComparison({products, preferredSkuId = null, dimensions = [], unverified = []}: {
  products: ProductCard[]; preferredSkuId?: string | null; dimensions?: string[]; unverified?: string[];
}) {
  const difference = comparisonDifference(products);
  const rows: {label: string; value: (p: ProductCard) => string}[] = [
    {label:"规格", value:p=>p.skus.find(s=>s.sku_id === p.default_sku_id)?.spec ?? "未指定"},
    {label:"库存", value:p=>{const sku=p.skus.find(s=>s.sku_id===p.default_sku_id);return sku?(sku.stock>0?"有货":"缺货"):"未提供";}},
    {label:"价格", value:p=>{const q=comparisonQuote(p);return q?`${money(q.total_amount_minor/100,q.currency)}（${(p.quantity??1)>1?`${p.quantity}件`:''}到手价）`:`${money(p.price_major,p.currency)}（商品价）`;}},
    {label:"适合与理由", value:p=>p.recommendation_reason ?? "未提供"},
    ...products.some(p=>p.highlights.length)?[{label:'已知特点',value:(p:ProductCard)=>p.highlights.join('；')||'—'}]:[],
    ...products.some(p=>p.weight_kg!=null)?[{label:'重量',value:(p:ProductCard)=>p.weight_kg!=null?`${p.weight_kg} kg`:'—'}]:[],
    ...products.some(p=>(p.quantity??1)>1)?[{label:'数量',value:(p:ProductCard)=>String(p.quantity??1)}]:[],
    ...products.some(p=>p.tradeoffs?.length)?[{label:'主要取舍',value:(p:ProductCard)=>p.tradeoffs?.join('；')||'—'}]:[],
    ...products.some(p=>p.constraint_issues?.length)?[{label:'当前限制',value:(p:ProductCard)=>p.constraint_issues?.map(i=>issueLabels[i]??i).join('；')||'—'}]:[],
  ];
  return <section aria-label="商品比较">
    <h2>商品比较</h2>
    {!!dimensions.length && <p className="comparison-focus">本次关注：{dimensions.join("、")}</p>}
    <div className="comparison-scroll"><table className="compare-table">
      <thead><tr><th>比较项</th>{products.map(p=><th key={comparisonKey(p)}>
        {p.image_url&&<ProductImage product={p}/>}<strong>{p.title}</strong><small>{p.source_platform?platformLabel(p.source_platform):p.skus.find(s=>s.sku_id===p.default_sku_id)?.spec}</small>
        {preferredSkuId === p.default_sku_id && <p>更推荐这款</p>}
      </th>)}</tr></thead>
      <tbody>{rows.map(row=><tr key={row.label}><td>{row.label}</td>{products.map(p=><td key={comparisonKey(p)}>{row.value(p)}</td>)}</tr>)}</tbody>
    </table></div>
    {difference&&<p>到手总价相差 {money(difference.amount,difference.currency)}</p>}
    {!!unverified.length && <p>尚未核验：{unverified.join("；")}</p>}
  </section>;
}
