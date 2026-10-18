import { memo, useState, type ReactNode } from "react";
import { comparisonKey, comparisonQuote } from "../lib/comparison";
import { availableOffers, groupProductRecords, platformLabel, skuPrice, type ProductGroup } from "../lib/productGroups";
import type { ProductCard } from "../types";
import Icon from "./Icon";
import {useProductMedia,ProductImageCaption,type MediaProduct} from "./CatalogMedia";
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
  onSelect?: (group: ProductGroup) => void;
  researchContent?: (product: ProductCard) => ReactNode;
}
interface RecommendationCardsProps extends CommonCardsProps {
  mode: "recommendation";
  preferredSkuId?: string | null;
  favoriteIds: Set<string>;
  comparedIds: Set<string>;
  onFavorite: (product: ProductCard) => void;
  onCompare: (product: ProductCard) => void;
}
type ProductCardsProps = RecommendationCardsProps | (CommonCardsProps & { mode: "view" | "research" });

function viewPrice(group: ProductGroup) {
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
  return <div className={`product-grid compact-product-grid ${entries.length===1?'product-grid--single':''}`}>
    {entries.map((group, index) => {
      const product = group.product;
      const recommendation = props.mode === "recommendation" ? props : null;
      const saved = recommendation?.favoriteIds.has(product.product_id) ?? false;
      const selected = recommendation?.comparedIds.has(comparisonKey(product)) ?? false;
      const primary = product.skus.find(sku => sku.sku_id === product.default_sku_id);
      const landed = recommendation ? comparisonQuote(product) : null;
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
            {product.rating_summary && <span className="rating">★ {product.rating_summary.average} <span>({product.rating_summary.review_count}{product.rating_is_live ? '' : ' · 样例'})</span></span>}
            {recommendation?.preferredSkuId != null && recommendation.preferredSkuId === product.default_sku_id
              && <strong className="preferred-pick">首选</strong>}
          </div>
          <button className="product-title" onClick={() => props.onDetail(group)}>{product.title}</button>
          <div className="product-subtitle">{(recommendation || props.mode === "research") ? (primary?.spec || product.highlights[0] || "查看商品详细信息")
            : [...new Set(group.records.map(p => platformLabel(p.source_platform)))].join(" · ")}</div>
          <div className="product-price-row">
            <div className="price">{recommendation
              ? money(landed ? landed.total_amount_minor / 100 : product.price_major, landed?.currency ?? product.currency) : props.mode === "research" ? (primary ? money(product.price_major,product.currency) : "规格待确认") : price.text}</div>
            <span className="price-kind">{recommendation ? (landed ? ((product.quantity ?? 1) > 1 ? `${product.quantity}件到手价` : "到手价") : "商品价") : props.mode === "research" ? "商品价" : price.kind}</span>
          </div>
          {(recommendation || props.mode === "research") && product.recommendation_reason && <p>{product.recommendation_reason}</p>}
          {recommendation&&!!product.tradeoffs?.length&&<p className="product-tradeoff">需要权衡：{product.tradeoffs[0]}</p>}
          {props.mode === "view" && product.highlights.length > 0 && <p>{product.highlights.slice(0, 2).join("；")}</p>}
          {props.mode === "research" && props.researchContent?.(product)}
          <div className="product-footer">
            {recommendation && <label className="compare-check"><input type="checkbox" checked={selected}
              onChange={() => recommendation.onCompare(product)} aria-label={`将 ${product.title} 加入比较`} />加入比较</label>}
            <button className="detail-button" onClick={() => props.onDetail(group)}>查看详情<Icon name="arrow" /></button>
            {props.mode !== "research" && <button className="select-product" onClick={()=> (props.onSelect??props.onDetail)(group)}>选这款</button>}
          </div>
        </div>
      </article>;
    })}
  </div>;
}

// 商品引用与交互回调不变时，不随助手逐字输出重绘商品区域。
export default memo(ProductCards);
