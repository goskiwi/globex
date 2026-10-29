/// <reference types="vite/client" />

export interface PriceLine {
  product_id: string;
  sku_id: string;
  title: string;
  quantity: number;
  unit_price_minor: number;
  source_unit_price_minor: number;
  source_currency: string;
  currency: string;
  subtotal_minor: number;
  freight_minor: number;
  tariff_minor: number;
  total_amount_minor: number;
}
export interface PriceQuote {
  items: PriceLine[];
  ship_to: string;
  currency: string;
  subtotal_minor: number;
  freight_minor: number;
  tariff_minor: number;
  total_amount_minor: number;
}
export type LandedPrice = PriceQuote | { unavailable_reason: string };

export interface ProductCard {
  product_id: string;
  title: string;
  brand: string;
  category: string;
  origin_country: string;
  price_major: number;
  currency: string;
  highlights: string[];
  skus: {
    sku_id: string;
    spec: string;
    price_major: number;
    currency: string;
    stock: number;
    display_price_major?: number;
    display_currency?: string;
    constraint_issues?: string[];
  }[];
  score: number;
  quantity?: number;
  recommendation_reason?: string;
  constraint_issues?: string[];
  tradeoffs?: string[];
  landed_price?: LandedPrice;
  description?: string;
  rating_summary?: { average: number; review_count: number } | null;
  rating_is_live?: boolean;
  ships_to?: string[];
  dimensions_cm?: { length?: number; width?: number; height?: number };
  package_dimensions_cm?: { length?: number; width?: number; height?: number };
  updated_at?: string;
  default_sku_id?: string;
  requested_sku_ids?: string[];
  selected_sku_id?: string;
  image_url?: string | null;
  image_kind?: "illustration" | "placeholder";
  image_alt?: string;
  source_platform?: string;
  source_price_major?: number;
  source_currency?: string;
  canonical_product_id?: string;
  material_tags?: string[];
  weight_kg?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  runId?: string;
}

export interface DiagnosticEvent {
  id: string;
  type: string;
  label: string;
  timestamp: number | null;
  detail?: string;
}

export type ProcessStatus = 'queued'|'running'|'completed'|'partial'|'failed'|'cancelled'|'interrupted'|'waiting_input'|'waiting_confirmation';
export type ProcessStepStatus = 'running'|'completed'|'partial'|'failed'|'cancelled'|'waiting_input'|'waiting_confirmation'|'unconfirmed';
export interface ProcessStep { id:string; label:string; status:ProcessStepStatus; summary:string }
export interface RunProcess { runId:string; userMessageId:string; status:ProcessStatus; steps:ProcessStep[]; error?:{code:string|null;message:string} }
export type ConnectionState = 'idle'|'connecting'|'connected'|'reconnecting'|'disconnected';

export interface SessionSummary {
  id: string;
  title: string;
  updatedAt: number;
  source?: "server" | "local";
}

export interface ToolApproval { id: string; tool: string; label: string; arguments: string | Record<string, unknown> }

export interface Comparison {
  guidance: string;
  hits: ProductCard[];
  preferred_sku_id: string | null;
  dimensions: string[];
  max_items: number;
  result_ref: string;
  unverified_requirements: string[];
}

export interface Recommendation {
  guidance: string;
  hits: ProductCard[];
  result_ref: string;
  unverified_requirements: string[];
  mode: "individual" | "bundle";
  quote: PriceQuote | null;
}


export interface ProductView {
  runId: string;
  guidance: string;
  hits: ProductCard[];
  result_ref: string;
  purpose: "product_view";
}

export interface Presentation {
  id:string; runId:string; kind:"recommendation"|"comparison"|"product_view"|"research"|"clarification"|"order";
  status:"completed"|"partial"|"needs_input"; payload:Record<string,unknown>;
}

export interface CommerceSnapshot {
  presentations: Presentation[];
  researchReports: ResearchDelivery[];
  productViews: ProductView[];
  comparison: Comparison | null;
  recommendation: Recommendation | null;
  productHistory?: {runId: string; updatedAt: number; products: ProductCard[]}[];
  shoppingForms: Record<string, any>[];
  shoppingFilters: Record<string, any>;
  toolApprovals?: ToolApproval[];
  skills: PublishedSkill[];
  skillsStatus: "loading" | "ready" | "error";
  skillsError: string | null;
  skillUsages: SkillUsage[];
  sessionId: string;
  messages: ChatMessage[];
  deliveredRunId: string | null;
  events: DiagnosticEvent[];
  processes: RunProcess[];
  connectionState: ConnectionState;
  stopRequested: boolean;
  status: "idle" | "running" | "stopped" | "error";
  error: string | null;
  history: SessionSummary[];
  confirmations: TradeConfirmation[];
  confirmationDecisions: ConfirmationDecision[];
  confirmationBusy: boolean;
  confirmationError: string | null;
  recoverableRunId: string | null;
  historyError: string | null;
}

export interface ShippingAddress {
  recipient_name: string;
  country: string;
  state: string;
  city: string;
  address_line: string;
  postal_code: string;
  phone: string;
}
export interface PrepareOrderInput {
  items: { product_id: string; sku_id: string; quantity: number }[];
  shipping_address: ShippingAddress;
  currency: string;
}
export interface OrderSnapshot {
  order_id: string;
  status: "CONFIRMED" | "CANCELLED";
  total_amount_minor: number;
  total_amount_major: number;
  currency: string;
  cancel_reason: string | null;
}
export interface TradeConfirmation {
  confirmation_id: string;
  operation_id: string;
  action: "create" | "cancel";
  buyer_id: string;
  session_id: string;
  snapshot_hash: string;
  expires_at: string;
  created_at: string;
  expired: boolean;
  status: "pending" | "approved" | "rejected";
  payload: {
    items: {
      product_id: string;
      sku_id: string;
      title: string;
      unit_price_minor: number;
      currency: string;
      quantity: number;
    }[];
    shipping_address: ShippingAddress;
    total_amount_minor: number;
    currency: string;
    subtotal_minor: number;
    freight_minor: number;
    tariff_minor: number;
    amount_scope: "landed";
    order_kind: "purchase_intent";
    order_id?: string;
    reason?: string;
    order_status?: string;
  };
  result: OrderSnapshot | null;
}

export type ConfirmationDecision = Pick<TradeConfirmation,
  'confirmation_id'|'buyer_id'|'session_id'|'action'|'status'|'expires_at'|'expired'>;

export interface PublishedSkill {
  id: string;
  version: string;
  title: string;
  description: string;
  scope: string;
  content_hash: string;
  expires_at: string | null;
}
export interface SkillUsage {
  toolCallId: string;
  id?: string;
  version?: string;
  title?: string;
  contentHash?: string;
  status: "reading" | "used" | "error";
}

export interface ResearchCandidate {
  product_id: string;
  sku_id: string | null;
  evidence_fields: string[];
  evidence: {field:string; label:string; value:unknown}[];
  missing_fields: {field:string; label:string}[];
  facts: {
    product: Partial<ProductCard>;
    sku: ProductCard['skus'][number] | null;
    historical: boolean;
    observed_at?: string;
    result_ref?: string;
    quotes: {ship_to: string; currency: string; historical: boolean; line: PriceLine}[];
  };
}
export interface ResearchReport {
  task_id: string;
  title: string;
  status: 'completed' | 'partial' | 'needs_input' | 'failed';
  product_count: number;
  sku_count: number;
  items: ResearchCandidate[];
  price_comparisons: {
    kind: 'unit_price' | 'landed_price'; quantity: number; currency: string; ship_to?: string;
    compared_sku_count: number; minimum_minor: number; maximum_minor: number; spread_minor: number;
    lowest: {product_id: string; sku_id: string; title: string; amount_minor: number}[];
    highest: {product_id: string; sku_id: string; title: string; amount_minor: number}[];
  }[];
  metric_comparisons: {
    field:string; label:string; unit:string; compared_product_count:number;
    excluded_products:{product_id:string; reason:'historical_only'|'missing_value'|'conflicting_observations'}[];
    minimum:number|null; maximum:number|null;
    lowest:{product_id:string; title:string; value:number}[];
    highest:{product_id:string; title:string; value:number}[];
  }[];
  unmet_requirements: string[];
  caveats: string[];
  questions: string[];
  issues: string[];
  transaction_state: 'none' | 'awaiting_confirmation';
}
export interface ResearchDelivery {
  runId: string;
  guidance: string;
  reports: ResearchReport[];
  status: 'completed' | 'partial';
  task_ids: string[];
  completed_task_ids: string[];
}


export interface OrderView {
  order_id:string; status:'DRAFT'|'CONFIRMED'|'CANCELLED'; currency:string; total_amount_minor:number;
  total_amount_major:number; created_at:string; shipping_address:string; cancel_reason:string|null;
  lines:{product_id:string;sku_id:string;title:string;unit_price_major:number;quantity:number}[];
}
export interface OrderOutcome {
  order:OrderView; observed_at:string; status:'completed'|'partial'; guidance:string;
  operation:{kind:'inspect'|'prepare_cancel';status:'completed'|'awaiting_confirmation'|'rejected';rejection:{code:string;message:string}|null};
  confirmation_required:boolean; confirmation:TradeConfirmation|null;
}
