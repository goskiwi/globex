export interface AuthSession { buyerId: string; accessToken: string; expiresAt: number }
export const AUTH_KEY = "globex.login.session";
export const apiBase = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export function readAuth(value: unknown): AuthSession | null {
  if (!value || typeof value !== "object") return null;
  const data=value as Record<string,unknown>;
  return typeof data.buyerId==="string" && !!data.buyerId && typeof data.accessToken==="string" && !!data.accessToken
    && typeof data.expiresAt==="number" && Number.isFinite(data.expiresAt) && data.expiresAt>Date.now()
    ? {buyerId:data.buyerId,accessToken:data.accessToken,expiresAt:data.expiresAt}:null;
}
