/// <reference types="vite/client" />

export type TradeEventType =
  | "agent.dispatch"
  | "tool.invoke"
  | "tool.result"
  | "plan.update"
  | "context.compressed"
  | "model.fallback"
  | "final.result"
  | "error";

export interface TradeEvent {
  type: TradeEventType;
  payload: Record<string, any>;
  occurred_at: string;
}

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
export interface RunProcess { runId:string; userMessageId:string; status:ProcessStatus; steps:ProcessStep[] }
export type ConnectionState = 'idle'|'connecting'|'connected'|'reconnecting'|'disconnected';

export interface SessionSummary {
  id: string;
  title: string;
  updatedAt: number;
  source?: "server" | "local";
}

export interface ToolApproval { id: string; tool: string; label: string; arguments: string | Record<string, unknown> }

export interface ProductDecision {
  guidance: string;
  hits: ProductCard[];
  preferred_sku_id: string | null;
  dimensions: string[];
  max_items: number;
  result_ref: string;
  unverified_requirements: string[];
}

export interface Recommendation extends ProductDecision { mode: "alternatives" | "bundle"; quote: PriceQuote | null }

export interface Comparison extends ProductDecision {}

export interface ProductView {
  runId: string;
  guidance: string;
  hits: ProductCard[];
  result_ref: string;
  purpose: "product_view";
}

export interface CommerceSnapshot {
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
