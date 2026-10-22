import type { TradeConfirmation, ConfirmationDecision } from "../types";
const record = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && !!v.trim();
const minor = (v: unknown): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
const currency = (v: unknown): v is string =>
  typeof v === "string" && /^[A-Z]{3}$/.test(v);

function compareCreated(a:TradeConfirmation,b:TradeConfirmation){
  // Date.parse 只保留毫秒；服务端时间含微秒，同毫秒内继续按原始小数部分排序。
  const milliseconds=Date.parse(b.created_at)-Date.parse(a.created_at);
  if(milliseconds)return milliseconds;
  const fraction=(value:string)=>(value.match(/\.(\d+)/)?.[1]??'').padEnd(9,'0');
  const first=fraction(a.created_at),second=fraction(b.created_at);
  return first<second?1:first>second?-1:a.confirmation_id<b.confirmation_id?1:a.confirmation_id>b.confirmation_id?-1:0;
}

/** 只有结构完整的服务端快照才能成为可点击的确认卡。 */
export function readConfirmations(value: unknown): TradeConfirmation[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is TradeConfirmation => {
      if (
        !record(item) ||
        !text(item.confirmation_id) ||
        !text(item.operation_id) ||
        !text(item.buyer_id) ||
        !text(item.session_id) ||
        !text(item.snapshot_hash) ||
        !text(item.expires_at) ||
        !text(item.created_at) || !Number.isFinite(Date.parse(item.created_at)) ||
        !Number.isFinite(Date.parse(item.expires_at)) ||
        !["create", "cancel"].includes(String(item.action)) ||
        !["pending", "approved", "rejected"].includes(String(item.status))
      )
        return false;
      const p = item.payload;
      if (
        !record(p) ||
        p.amount_scope !== "landed" ||
        p.order_kind !== "purchase_intent" ||
        !minor(p.total_amount_minor) || !minor(p.subtotal_minor) || !minor(p.freight_minor) || !minor(p.tariff_minor) ||
        !currency(p.currency) ||
        !Array.isArray(p.items) ||
        !p.items.length
      )
        return false;
      let total = 0;
      for (const line of p.items) {
        if (
          !record(line) ||
          !text(line.product_id) ||
          !text(line.sku_id) ||
          !text(line.title) ||
          !minor(line.unit_price_minor) ||
          line.currency !== p.currency ||
          !minor(line.quantity) ||
          line.quantity <= 0
        )
          return false;
        total += line.unit_price_minor * line.quantity;
      }
      if (!Number.isSafeInteger(total) || total !== p.subtotal_minor || total + Number(p.freight_minor) + Number(p.tariff_minor) !== p.total_amount_minor)
        return false;
      const address = p.shipping_address;
      if (
        !record(address) ||
        !["recipient_name", "country", "city", "address_line"].every((key) =>
          text(address[key]),
        ) ||
        !["state", "postal_code", "phone"].every(
          (key) => typeof address[key] === "string",
        )
      )
        return false;
      if (item.action === "cancel" && (!text(p.order_id) || !text(p.reason)))
        return false;
      if (item.status === "approved") {
        const result = item.result;
        if (
          !record(result) ||
          !text(result.order_id) ||
          !["CONFIRMED", "CANCELLED"].includes(String(result.status)) ||
          !minor(result.total_amount_minor) ||
          !currency(result.currency)
        )
          return false;
      } else if (item.result !== null) return false;
      return true;
    })
    .map((item) => ({
      ...item,
      expired:
        item.status === 'pending' && (item.expired === true || Date.parse(item.expires_at) <= Date.now()),
    }));
}

/** 决议单向推进，迟到的流式 pending 快照不能让确认按钮重新出现。 */
export function mergeConfirmations(
  previous: TradeConfirmation[],
  next: TradeConfirmation[],
): TradeConfirmation[] {
  const entries = new Map(previous.map((item) => [item.confirmation_id, item]));
  for (const item of next) {
    const old = entries.get(item.confirmation_id);
    entries.set(
      item.confirmation_id,
      old && old.status !== "pending" ? old : item,
    );
  }
  const all = [...entries.values()].sort(compareCreated);
  const now = Date.now();
  const active = new Set(all.filter(item => item.status === "pending" && !item.expired
    && Date.parse(item.expires_at) > now).map(item => item.confirmation_id));
  const history = new Set(all.filter(item => !active.has(item.confirmation_id)).slice(0,20)
    .map(item => item.confirmation_id));
  return all
    .filter(item => active.has(item.confirmation_id) || history.has(item.confirmation_id));
}

export function readConfirmationDecisions(value:unknown):ConfirmationDecision[]{
 if(!Array.isArray(value))return [];
 return value.filter((d):d is ConfirmationDecision=>record(d)
  &&['confirmation_id','buyer_id','session_id'].every(k=>text(d[k]))
  &&['create','cancel'].includes(String(d.action))&&['pending','approved','rejected'].includes(String(d.status))
  &&text(d.expires_at)&&Number.isFinite(Date.parse(d.expires_at))&&typeof d.expired==='boolean')
  .map(d=>({confirmation_id:d.confirmation_id,buyer_id:d.buyer_id,session_id:d.session_id,action:d.action,
   status:d.status,expires_at:d.expires_at,expired:d.status==='pending'&&(d.expired||Date.parse(d.expires_at)<=Date.now())}));
}

export function mergeConfirmationDecisions(previous:ConfirmationDecision[],incoming:ConfirmationDecision[]):ConfirmationDecision[]{
 const entries=new Map(previous.map(c=>[c.confirmation_id,c]));
 for(const d of incoming){const old=entries.get(d.confirmation_id);
  if(!old||old.status==='pending')entries.set(d.confirmation_id,d);
 }
 return [...entries.values()];
}
