import type {ResearchDelivery} from '../types';

const record = (v: unknown): v is Record<string, any> => !!v && typeof v === 'object' && !Array.isArray(v);
const strings = (v: unknown): v is string[] => Array.isArray(v) && v.every(x => typeof x === 'string');
const amount = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0;

/** 读取服务端研究投影；不从历史正文重建候选、价格或取舍。 */
export function readResearchReports(value: unknown): ResearchDelivery[] {
  if (!Array.isArray(value)) return [];
  return value.filter(batch => record(batch) && typeof batch.runId === 'string' && batch.runId
    && typeof batch.guidance === 'string' && ['completed','partial'].includes(batch.status)
    && strings(batch.task_ids) && strings(batch.completed_task_ids) && Array.isArray(batch.reports)
    && batch.reports.every((r: unknown) => record(r) && typeof r.task_id === 'string' && typeof r.goal === 'string'
      && ['completed','partial','needs_input','failed'].includes(r.status) && amount(r.product_count) && amount(r.sku_count)
      && ['unmet_requirements','caveats','questions','issues'].every(k=>strings(r[k]))
      && Array.isArray(r.items) && r.items.every((c: unknown)=>record(c) && typeof c.product_id === 'string'
        && (c.sku_id === null || typeof c.sku_id === 'string') && typeof c.assessment === 'string'
        && strings(c.caveats) && record(c.facts) && record(c.facts.product)
        && typeof c.facts.historical === 'boolean' && Array.isArray(c.facts.quotes)
        && (c.facts.sku === null || record(c.facts.sku) && typeof c.facts.sku.spec === 'string'
          && amount(c.facts.sku.price_major) && typeof c.facts.sku.currency === 'string'))
      && Array.isArray(r.price_comparisons) && r.price_comparisons.every((p: unknown)=>record(p)
        && ['unit_price','landed_price'].includes(p.kind) && typeof p.currency === 'string'
        && ['minimum_minor','maximum_minor','spread_minor','quantity','compared_sku_count'].every(k=>amount(p[k]))
        && ['lowest','highest'].every(k=>Array.isArray(p[k]) && p[k].every((v: unknown)=>record(v) && typeof v.title === 'string'))))) as ResearchDelivery[];
}
