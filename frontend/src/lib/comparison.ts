import type {ProductCard, PriceQuote} from "../types";

export const comparisonKey = (product: ProductCard) => `${product.product_id}:${product.default_sku_id ?? ""}`;

export function comparisonQuote(product: ProductCard): PriceQuote | null {
  const quote = product.landed_price;
  if (!quote || "unavailable_reason" in quote || quote.items.length !== 1) return null;
  const line = quote.items[0];
  return line.product_id === product.product_id && line.sku_id === product.default_sku_id &&
    line.quantity === (product.quantity ?? 1) && quote.currency === product.currency ? quote : null;
}

export function comparisonDifference(products: ProductCard[]): {amount: number; currency: string} | null {
  if (products.length < 2) return null;
  const quotes = products.map(comparisonQuote);
  const first = quotes[0];
  if (!first || quotes.some(q => !q || q.currency !== first.currency || q.ship_to !== first.ship_to ||
    q.items[0].quantity !== first.items[0].quantity)) return null;
  const totals = quotes.map(q => q!.total_amount_minor);
  return {amount: (Math.max(...totals) - Math.min(...totals)) / 100, currency: first.currency};
}

export function upsertComparison(products: ProductCard[], product: ProductCard): ProductCard[] {
  const key = comparisonKey(product);
  return products.some(p => comparisonKey(p) === key)
    ? products.map(p => comparisonKey(p) === key ? product : p) : [...products, product];
}
