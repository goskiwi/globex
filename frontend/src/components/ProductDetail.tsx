import { useState } from "react";
import type { ProductCard } from "../types";
import { initialOffer, platformLabel, productFactKeys, skuPrice, type ProductGroup } from "../lib/productGroups";
import { comparisonQuote } from "../lib/comparison";
import Icon from "./Icon";
import Modal from "./Modal";
import { money, ProductImage } from "./ProductCards";
import ProductFacts from "./ProductFacts";

export default function ProductDetail({ group, busy, onClose, onCompare, onAsk, onPrepare }: {
  group: ProductGroup;
  busy: boolean;
  onClose: () => void;
  onCompare: (product: ProductCard) => void;
  onAsk: (query: string) => void;
  onPrepare: (product: ProductCard, skuId: string, currency: string) => void;
}) {
  const [selection, setSelection] = useState(() => {
    const offer = initialOffer(group);
    return { productId: offer?.product.product_id ?? group.records[0].product_id, skuId: offer?.sku.sku_id ?? "" };
  });
  const product = group.records.find(p => p.product_id === selection.productId)!;
  const sku = product.skus.find(s => s.sku_id === selection.skuId);
  const quote = comparisonQuote(product);
  const landed = sku?.sku_id === product.default_sku_id ? quote : null;
  const price = landed ? { amount: landed.items[0].unit_price_minor / 100, currency: landed.currency } : sku ? skuPrice(sku) : null;
  const overBudget = sku?.constraint_issues?.some(issue => ["over_price_cap", "over_landed_budget"].includes(issue));

  function selectPlatform(record: ProductCard) {
    const next = record.skus.find(s => s.sku_id === record.default_sku_id && s.stock > 0)
      ?? record.skus.find(s => s.stock > 0) ?? record.skus.find(s => s.sku_id === record.default_sku_id) ?? record.skus[0];
    setSelection({ productId: record.product_id, skuId: next?.sku_id ?? "" });
  }

  return <Modal title={`${group.product.title} 商品详情`} drawer onClose={onClose}>
    {group.product.image_url&&<div className="drawer-visual"><ProductImage product={group.product} /></div>}
    <div className="drawer-kicker">{group.product.brand}</div>
    <h2>{group.product.title}</h2>
    <p className="drawer-description">{group.product.description ?? product.description ?? product.highlights.join("；")}</p>
    {product.recommendation_reason&&<p className="drawer-description">{product.recommendation_reason}</p>}
    {!!product.tradeoffs?.length&&<ul className="drawer-tradeoffs">{product.tradeoffs.map(item=><li key={item}>{item}</li>)}</ul>}
    <ProductFacts product={group.product} fields={group.sharedFields} offerFields={false} />
    <fieldset className="sku-picker platform-picker">
      <legend>购买平台</legend>
      {group.records.map(record => <label key={record.product_id} className={product.product_id === record.product_id ? "chosen" : ""}>
        <input type="radio" name="platform" value={record.product_id} checked={product.product_id === record.product_id}
          onChange={() => selectPlatform(record)} /><span>{platformLabel(record.source_platform)}</span>
      </label>)}
    </fieldset>
    <fieldset className="sku-picker">
      <legend>选择规格</legend>
      {product.skus.map(item => {
        const amount = skuPrice(item);
        return <label key={item.sku_id} className={selection.skuId === item.sku_id ? "chosen" : ""}>
          <input type="radio" name="product-sku" value={item.sku_id} checked={selection.skuId === item.sku_id}
            onChange={() => setSelection({ productId: product.product_id, skuId: item.sku_id })} />
          <span>{item.spec}</span><strong>{money(amount.amount, amount.currency)}</strong>
          <small>{item.stock > 0 ? "有货" : "缺货"}</small>
        </label>;
      })}
    </fieldset>
    {price && <><div className="price">{money(price.amount, price.currency)}</div><span className="detail-price-kind">当前规格商品价</span></>}
    {sku && price && price.currency !== sku.currency && <p className="platform-rating">原价 {money(sku.price_major, sku.currency)}</p>}
    {product.rating_summary && <p className="platform-rating">{platformLabel(product.source_platform)} 评分 {product.rating_summary.average}（{product.rating_summary.review_count}条）</p>}
    <ProductFacts product={product} fields={productFactKeys.filter(key => !group.sharedFields.includes(key))} offerFields={false} />
    {overBudget && <p className="drawer-budget-note">超出当前选购预算。</p>}
    {landed ? <div className="landed-detail-panel">
      <strong>到手价 {money(landed.total_amount_minor / 100, landed.currency)}</strong>
      <span>小计 {money(landed.subtotal_minor / 100, landed.currency)} + 运费 {money(landed.freight_minor / 100, landed.currency)} + 关税 {money(landed.tariff_minor / 100, landed.currency)}</span>
      <small>配送至 {landed.ship_to} · {product.quantity || 1} 件</small>
    </div> : <p className="drawer-note">到手价将在填写配送信息后计算。</p>}
    <div className="drawer-actions"><button className="drawer-compare" disabled={busy || !sku} onClick={() => {
      onAsk(`请进一步核对「${product.title}」（product_id=${product.product_id}，sku_id=${sku!.sku_id}，规格=${sku!.spec}）的当前库存与到手价。`);
      onClose();
    }}><Icon name="chat" />{busy ? "正在处理上一条需求" : "帮我进一步确认这款"}</button>
    <button className="primary-button" disabled={busy || !sku || sku.stock <= 0}
      onClick={() => sku && price && onPrepare(product, sku.sku_id, price.currency)}>{sku && sku.stock <= 0 ? "当前规格缺货" : "准备下单"}</button>
    <button className="drawer-compare" disabled={!sku} onClick={() => sku && price && onCompare({
      ...product, default_sku_id: sku.sku_id, price_major: price.amount, currency: price.currency,
      source_price_major: sku.price_major, source_currency: sku.currency, landed_price: landed ?? undefined,
      recommendation_reason: sku.sku_id === product.default_sku_id ? product.recommendation_reason : undefined,
      tradeoffs: sku.sku_id === product.default_sku_id ? product.tradeoffs : undefined,
      constraint_issues: sku.constraint_issues,
    })}><Icon name="compare" />加入比较</button></div>
  </Modal>;
}
