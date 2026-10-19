import { memo, useState } from "react";
import { comparisonKey, comparisonQuote } from "../lib/comparison";
import { availableOffers, groupProductRecords, initialOffer, platformLabel, skuPrice, type ProductGroup } from "../lib/productGroups";
import type { ProductCard } from "../types";
import Icon from "./Icon";
import {useProductMedia,ProductImageCaption,type MediaProduct} from "./CatalogMedia";
import "./originalProductCard.css";
import "./compactProductCards.css";

export function money(value: number, currency: string) {
  if (!Number.isFinite(value)) return "待确认";
  try {
    return new Intl.NumberFormat("zh-CN", { style: "currency", currency, maximumFractionDigits: 2 }).format(value);
  } catch {
    return `${value} ${currency}`;
  }
}

export function ProductImage({
  product,
  className = "",
}: {
  product: MediaProduct;
  className?: string;
}) {
  const media = useProductMedia(product);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  // 服务端注册表按精确 ID 补图，不从标题猜测；URL 更新后允许新图片重试。
  if (!media.url || failedUrl === media.url)
    return (
      <div className={`image-placeholder ${className}`}>
        <Icon name="bag" />
        <span>{product.category || "好物详情"}</span>
        <small>暂未提供商品图片</small>
      </div>
    );
  if (media.cell !== undefined) return <svg className={`catalog-media-image ${className}`} viewBox="0 0 1 1" role="img" aria-label={media.alt}>
    <image href={media.url} x={-(media.cell % 4)} y={-Math.floor(media.cell / 4)} width="4" height="4" onError={() => setFailedUrl(media.url)} />
  </svg>;
  return (
    <img
      className={className}
      src={media.url}
      alt={media.alt}
      loading="lazy"
      onError={() => setFailedUrl(media.url)}
    />
  );
}
interface CommonCardsProps {
  products: ProductCard[];
  onDetail: (group: ProductGroup) => void;
}
interface RecommendationCardsProps extends CommonCardsProps {
  mode: "recommendation";
  favoriteIds: Set<string>;
  comparedIds: Set<string>;
  onFavorite: (product: ProductCard) => void;
  onCompare: (product: ProductCard) => void;
}
type ProductCardsProps = RecommendationCardsProps | (CommonCardsProps & { mode: "view" | "research" });

function viewPrice(group: ProductGroup) {
  const focused=group.initialSkuId ? initialOffer(group) : undefined;
  if(focused){const price=skuPrice(focused.sku);return {text:money(price.amount,price.currency),kind:'商品价'};}

  const offers = availableOffers(group);
  if (!offers.length) return { text: "当前缺货", kind: "" };
  const prices = offers.map(o => skuPrice(o.sku));
  if (!prices.every(p => p.currency === prices[0].currency)) return { text: "查看各平台报价", kind: "" };
  const low = Math.min(...prices.map(p => p.amount)), high = Math.max(...prices.map(p => p.amount));
  return { text: money(low, prices[0].currency), kind: low === high ? "商品价" : "商品价起" };
}

function ProductCards(props: ProductCardsProps) {
  const entries = props.mode === "view" ? groupProductRecords(props.products)
    : props.products.map(product => ({ ...groupProductRecords([product])[0], key: comparisonKey(product), initialSkuId: product.default_sku_id }));
  return <div className="product-grid compact-product-grid">
    {entries.map((group, index) => {
      const product = group.product;
      const recommendation = props.mode === "recommendation" ? props : null;
      const saved = recommendation?.favoriteIds.has(product.product_id) ?? false;
      const selected = recommendation?.comparedIds.has(comparisonKey(product)) ?? false;
      const focused = props.mode === 'view' ? initialOffer(group) : undefined;
      const primary = props.mode === 'view' ? focused?.sku
        : product.skus.find(sku=>sku.sku_id===product.default_sku_id);
      const landed = props.mode === 'view'
        ? focused ? comparisonQuote({...focused.product, default_sku_id:focused.sku.sku_id}) : null
        : comparisonQuote(product);
      const price = viewPrice(group);
      return <article className={`product-card ${selected ? "selected" : ""}`} key={group.key}
        data-card-mode={props.mode} style={{ animationDelay: `${Math.min(index, 5) * 55}ms` }}>
        {<div className="product-visual">
          <button className="image-open" onClick={() => props.onDetail(group)} aria-label={`查看 ${product.title} 详情`}>
            <ProductImage product={product} />
          </button>
          <span className="product-badge">{product.category}</span>
          <span className="visual-caption"><ProductImageCaption product={product}/></span>
          {recommendation && <button className={`heart ${saved ? "saved" : ""}`} onClick={() => recommendation.onFavorite(product)}
            aria-label={`${saved ? "取消收藏" : "收藏"} ${product.title}`} aria-pressed={saved}><Icon name="heart" /></button>}
        </div>}
        <div className="product-body">
          <div className="product-topline"><span>{product.brand || "精选商品"}</span>
            {product.rating_summary && <span className="rating"><i>★</i> {product.rating_summary.average.toFixed(1)} <span>({product.rating_summary.review_count}{product.rating_is_live ? '' : ' · 样例'})</span></span>}
          </div>
          <button className="product-title" onClick={() => props.onDetail(group)}>{product.title}</button>
          <div className="product-subtitle">{(recommendation || props.mode === "research" || !!group.initialSkuId) ? (primary?.spec || product.highlights[0] || "查看商品详细信息")
            : [...new Set(group.records.map(p => platformLabel(p.source_platform)))].join(" · ")}</div>
          <div className="product-price-row">
            <div className="price">{props.mode === "view" ? price.text : primary ? money(product.price_major,product.currency) : "规格待确认"}</div>
            <span className="price-kind">{props.mode === "view" ? price.kind : "商品价"}</span>
          </div>
          {landed ? <div className="card-landed">
            到手价 {money(landed.total_amount_minor/100,landed.currency)}
            <span>配送至 {landed.ship_to} · {landed.items[0].quantity===1 ? '当前规格' : `${landed.items[0].quantity}件`}</span>
          </div> : <div className="card-landed pending">到手价待目的地与规格确认</div>}
          <div className="product-footer">
            {recommendation && <label className="compare-check"><input type="checkbox" checked={selected}
              onChange={() => recommendation.onCompare(product)} aria-label={`将 ${product.title} 加入比较`} />加入比较</label>}
            <button className="detail-button" onClick={() => props.onDetail(group)}>看看细节<Icon name="arrow" /></button>
          </div>
        </div>
      </article>;
    })}
  </div>;
}

// 商品引用与交互回调不变时，不随助手逐字输出重绘商品区域。
export default memo(ProductCards);
