import type {ProductCard} from '../types';

export const productFactKeys=['category','origin_country','material_tags','weight_kg','dimensions_cm','ships_to'] as const;
export type ProductFactKey=typeof productFactKeys[number];
export interface ProductGroup { key:string; product:ProductCard; records:ProductCard[]; sharedFields:ProductFactKey[]; initialSkuId?:string }
export type ProductSku = ProductCard['skus'][number];
export interface ProductOffer { product:ProductCard; sku:ProductSku }

export function platformLabel(platform?:string):string {
  return ({amazon:'Amazon',ebay:'eBay'} as Record<string,string>)[platform??'']??platform??'商品';
}

export function skuPrice(sku:ProductSku) {
  return {amount:sku.display_price_major??sku.price_major,currency:sku.display_currency??sku.currency};
}

export function availableOffers(group:ProductGroup):ProductOffer[] {
  const stocked=group.records.flatMap(product=>product.skus.filter(sku=>sku.stock>0).map(sku=>({product,sku})));
  const deliverable=stocked.filter(({product,sku})=>!(sku.constraint_issues??product.constraint_issues??[]).includes('ship_to_unavailable'));
  // 已知目的地不匹配的记录仍可查看；只调整购买起点，不删平台和规格，也不按预算屏蔽详情。
  return deliverable.length?deliverable:stocked;
}

/** 默认选择保留推荐规格；同款查看优先选择有货报价，不让缺货低价成为购买起点。 */
export function initialOffer(group:ProductGroup):ProductOffer|undefined {
  if(group.initialSkuId){
    for(const product of group.records){
      const sku=product.skus.find(s=>s.sku_id===group.initialSkuId);
      if(sku)return {product,sku};
    }
  }
  const selected=group.records.flatMap(product=>product.skus
    .filter(sku=>sku.sku_id===product.selected_sku_id).map(sku=>({product,sku})));
  if(selected.length===1)return selected[0];
  const offers=availableOffers(group);
  if(offers.length){
    const currency=skuPrice(offers[0].sku).currency;
    return offers.every(o=>skuPrice(o.sku).currency===currency)
      ?offers.reduce((best,o)=>skuPrice(o.sku).amount<skuPrice(best.sku).amount?o:best):offers[0];
  }
  const product=group.records[0],sku=product?.skus.find(s=>s.sku_id===product.default_sku_id)??product?.skus[0];
  return sku?{product,sku}:undefined;
}

/** 只按目录同款主键分组；标题相同不构成同款证据，平台记录/SKU原样保留。 */
export function groupProductRecords(products:ProductCard[]):ProductGroup[]{
  const groups=new Map<string,ProductCard[]>();
  for(const product of products){
    const key=product.canonical_product_id?`canonical:${product.canonical_product_id}`:`record:${product.product_id}`;
    groups.set(key,[...(groups.get(key)??[]),product]);
  }
  return [...groups].map(([key,records])=>{
    const first=records[0];
    const sharedFields=productFactKeys.filter(field=>records.every(p=>JSON.stringify(p[field])===JSON.stringify(first[field])));
    const description=records.every(p=>p.description===first.description)?first.description:undefined;
    const image=records.find(p=>p.image_url);
    return {key,records,sharedFields,product:{...first,description,
      image_url:image?.image_url,image_alt:image?.image_alt,image_kind:image?.image_kind}};
  });
}
