import {useState} from 'react';
import type {ResearchDelivery, ResearchCandidate, ProductCard, PriceQuote} from '../types';
import ProductCards, {money} from './ProductCards';
import Modal from './Modal';
import './researchReports.css';

const labels: Record<string,string> = {length:'长',width:'宽',height:'高',average:'评分',review_count:'评价数'};
const detailFields: {field: keyof ProductCard; label: string}[] = [
  {field:'highlights',label:'目录亮点'}, {field:'dimensions_cm',label:'商品尺寸（厘米）'},
  {field:'material_tags',label:'目录材质'}, {field:'rating_summary',label:'目录评分'},
  {field:'ships_to',label:'目录配送范围'},
];
function factText(value: unknown): string {
  if (Array.isArray(value)) return value.map(factText).join('；');
  if (value && typeof value === 'object') return Object.entries(value).map(([k,v])=>`${labels[k] ?? k} ${factText(v)}`).join(' · ');
  return value == null ? '未提供' : String(value);
}

/** 把当前研究证据投影到共用卡片；不查询新库存、不补猜图片或规格。 */
function card(candidate: ResearchCandidate): ProductCard {
  const {product,sku}=candidate.facts;
  const currency=sku?.display_currency ?? '';
  const quotes=candidate.facts.historical ? [] : candidate.facts.quotes.filter(q=>!q.historical && q.currency===currency
    && q.line.currency===currency && q.line.product_id===candidate.product_id && q.line.sku_id===candidate.sku_id && q.line.quantity===1);
  const read=quotes.length===1 ? quotes[0] : null;
  const quote: PriceQuote|undefined=read ? {...read,items:[read.line],subtotal_minor:read.line.subtotal_minor,
    freight_minor:read.line.freight_minor,tariff_minor:read.line.tariff_minor,total_amount_minor:read.line.total_amount_minor} : undefined;
  return {...product,product_id:candidate.product_id,title:product.title ?? candidate.product_id,
    brand:product.brand ?? '',category:product.category ?? '',origin_country:product.origin_country ?? '',
    highlights:product.highlights ?? [],score:0,skus:sku?[sku]:[],default_sku_id:sku?.sku_id,
    price_major:sku?.display_price_major ?? Number.NaN,
    currency,landed_price:quote,recommendation_reason:candidate.assessment};
}

function Questions({candidate}: {candidate: ResearchCandidate}) {
  return <>
    {!!candidate.caveats.length && <details className="research-questions"><summary>待核验事项 · {candidate.caveats.length}</summary>
      <ul>{candidate.caveats.map((question,i)=><li key={i}>{question}</li>)}</ul></details>}
  </>;
}

/** 与推荐复用商品卡，研究候选不显示首选、选购、比较或下单操作。 */
export default function ResearchReports({delivery}: {delivery: ResearchDelivery}) {
  const [detail,setDetail]=useState<ResearchCandidate|null>(null);
  return <section className="research-reports" aria-label="研究候选与取舍">
    {delivery.reports.map(report=><section className="research-category" key={report.task_id} aria-label={report.goal}>
      <div className="research-heading"><h3>{report.goal}</h3><span>{report.status==='completed'?'研究完成':report.status==='partial'?'部分完成':report.status==='needs_input'?'需要补充信息':'研究未完成'}</span></div>
      <p className="research-count">{report.product_count}款商品 · {report.sku_count}个明确规格</p>
      {!!report.items.length && <div className="research-analysis" aria-label="选择依据与取舍">
        {report.items.map(candidate=><div key={`${candidate.product_id}:${candidate.sku_id}`}>
          <p><strong>{candidate.facts.product.title ?? candidate.product_id}：</strong>{candidate.assessment}</p>
          {candidate.facts.historical && <p>历史资料，尚未重新核验。</p>}
        </div>)}
      </div>}
      {!!report.caveats.length && <p>资料局限：{report.caveats.join('；')}</p>}
      {!!report.unmet_requirements.length && <p>未满足明确要求：{report.unmet_requirements.join('；')}</p>}
      <ProductCards mode="research" products={report.items.map(card)}
        onDetail={group=>setDetail(report.items.find(c=>c.product_id===group.product.product_id && (c.sku_id ?? undefined)===group.initialSkuId)!)}/>
      {!report.items.length && <p>本次没有交回符合要求的候选，未满足条件见上方说明。</p>}
      {!!report.price_comparisons.length && <details className="research-prices"><summary>同口径价格比较</summary>
        {report.price_comparisons.map((price,i)=><div key={i}>
          <p>{price.kind==='unit_price'?'单件商品价，不含运税':`每个规格${price.quantity}件到手报价，配送至${price.ship_to}`} · 本次{price.compared_sku_count}个规格</p>
          <dl><div><dt>最低</dt><dd>{money(price.minimum_minor/100,price.currency)} · {price.lowest.map(p=>p.title).join('、')}</dd></div>
            <div><dt>最高</dt><dd>{money(price.maximum_minor/100,price.currency)} · {price.highest.map(p=>p.title).join('、')}</dd></div>
            <div><dt>差额</dt><dd>{money(price.spread_minor/100,price.currency)}</dd></div></dl>
        </div>)}
      </details>}
    </section>)}
    <p className="research-caption">以上为当时读取的资料与研究建议。价格、库存与未明确的适配能力，购买前仍需核验。</p>
    {detail && <Modal title={`${detail.facts.product.title ?? detail.product_id} 已读资料`} drawer onClose={()=>setDetail(null)}>
      <h2>{detail.facts.product.title ?? detail.product_id}</h2>
      <p>{detail.assessment}</p>
      <Questions candidate={detail}/>
      <div className="research-source">
        {detail.facts.sku && <p>已读规格：{detail.facts.sku.spec} · 原币商品价 {money(detail.facts.sku.price_major,detail.facts.sku.currency)}</p>}
        {detail.facts.product.weight_kg != null && <p>重量：{detail.facts.product.weight_kg} kg</p>}
        {detail.facts.historical && <p>历史资料，尚未重新核验。</p>}
        {detail.facts.product.description && <p>{detail.facts.product.description}</p>}
        {detailFields.map(({field,label})=> {
          const value=detail.facts.product[field];
          return value == null || value === '' || typeof value === 'object' && !Object.keys(value).length
            ? null : <p key={field}><strong>{label}：</strong>{factText(value)}</p>;
        })}
        {detail.facts.product.updated_at && <p>目录日期：{detail.facts.product.updated_at}</p>}
        {detail.facts.observed_at && <p>读取时间：{detail.facts.observed_at}</p>}
        {detail.facts.quotes.map((quote,i)=><p key={i}>{quote.historical ? '历史' : '已读'}到手报价：{quote.line.quantity}件，配送至{quote.ship_to}，{money(quote.line.total_amount_minor/100,quote.currency)}</p>)}
      </div>
    </Modal>}
  </section>;
}
