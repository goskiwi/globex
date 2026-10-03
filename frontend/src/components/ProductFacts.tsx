import type {ProductCard} from '../types';
import type {ProductFactKey} from '../lib/productGroups';

export default function ProductFacts({product,fields,offerFields=true}:{product:ProductCard;fields?:readonly ProductFactKey[];offerFields?:boolean}){
  const size=product.dimensions_cm;
  const facts:[ProductFactKey,string,string|undefined][]=[
    ['category','商品分类',product.category],['origin_country','原产地',product.origin_country],
    ['material_tags','材质',product.material_tags?.join('、')],
    ['weight_kg','重量',product.weight_kg?`${product.weight_kg} kg`:undefined],
    ['dimensions_cm','尺寸',size?.length&&size.width&&size.height?`${size.length} × ${size.width} × ${size.height} cm`:undefined],
    ['ships_to','配送地区',product.ships_to?.join(' / ')],
  ];
  const rows=facts.filter(([key,,value])=>value&&(!fields||fields.includes(key))).map(([,label,value])=>[label,value]);
  if(offerFields){
    if(product.source_platform)rows.unshift(['平台',product.source_platform]);
    if(product.rating_summary)rows.push(['评分',`${product.rating_summary.average}（${product.rating_summary.review_count}条）`]);
  }
  if(!rows.length)return null;
  return <div className="detail-specs">{rows.map(([label,value])=><div key={label}><span>{label}</span><span>{value}</span></div>)}</div>;
}
