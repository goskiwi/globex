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
    && batch.reports.every((r: unknown) => record(r) && typeof r.task_id === 'string' && typeof r.title === 'string' && r.title.trim().length > 0 && Array.from(r.title).length <= 80
      && ['completed','partial','needs_input','failed'].includes(r.status) && amount(r.product_count) && amount(r.sku_count)
      && ['none','awaiting_confirmation'].includes(r.transaction_state)
      && ['unmet_requirements','caveats','questions','issues'].every(k=>strings(r[k]))
      && Array.isArray(r.items) && r.items.every((c: unknown)=>record(c) && typeof c.product_id === 'string'
        && (c.sku_id === null || typeof c.sku_id === 'string') && strings(c.evidence_fields)
        && Array.isArray(c.evidence) && c.evidence.every((e:unknown)=>record(e) && typeof e.field==='string' && typeof e.label==='string' && 'value' in e)
        && Array.isArray(c.missing_fields) && c.missing_fields.every((e:unknown)=>record(e) && typeof e.field==='string' && typeof e.label==='string')
        && record(c.facts) && record(c.facts.product)
        && typeof c.facts.historical === 'boolean' && Array.isArray(c.facts.quotes)
        && (c.facts.sku === null || record(c.facts.sku) && typeof c.facts.sku.spec === 'string'
          && amount(c.facts.sku.price_major) && typeof c.facts.sku.currency === 'string'))
      && Array.isArray(r.metric_comparisons) && r.metric_comparisons.every((p:unknown)=>record(p)
        && ['field','label','unit'].every(k=>typeof p[k]==='string')
        && amount(p.compared_product_count) && ['minimum','maximum'].every(k=>p[k]===null||amount(p[k]))
        && Array.isArray(p.excluded_products) && p.excluded_products.every((e:unknown)=>record(e)
          && typeof e.product_id==='string' && ['historical_only','missing_value','conflicting_observations'].includes(e.reason))
        && ['lowest','highest'].every(k=>Array.isArray(p[k]) && p[k].every((v:unknown)=>record(v) && typeof v.product_id==='string' && typeof v.title==='string' && amount(v.value))))
      && Array.isArray(r.price_comparisons) && r.price_comparisons.every((p: unknown)=>record(p)
        && ['unit_price','landed_price'].includes(p.kind) && typeof p.currency === 'string'
        && ['minimum_minor','maximum_minor','spread_minor','quantity','compared_sku_count'].every(k=>amount(p[k]))
        && ['lowest','highest'].every(k=>Array.isArray(p[k]) && p[k].every((v: unknown)=>record(v) && typeof v.title === 'string'))))) as ResearchDelivery[];
}
