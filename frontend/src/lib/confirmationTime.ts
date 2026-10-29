import {useSyncExternalStore} from 'react';
import type {ConfirmationDecision} from '../types';

/** 确认卡和历史成果共用有效期语义，用户决议终态优先。 */
export function isConfirmationExpired(confirmation: Pick<ConfirmationDecision,'status'|'expires_at'|'expired'>, now: number) {
  if (confirmation.status !== 'pending') return false;
  const expiresAt = Date.parse(confirmation.expires_at);
  return confirmation.expired || !Number.isFinite(expiresAt) || expiresAt <= now;
}

let now = Date.now();
let timer: ReturnType<typeof setInterval> | undefined;
const listeners = new Set<() => void>();
const snapshot = () => now;
const serverSnapshot = () => Date.now();
function subscribe(listener: () => void) {
  listeners.add(listener);
  if (timer === undefined) {
    now = Date.now();
    timer = setInterval(() => {
      now = Date.now();
      listeners.forEach(notify => notify());
    }, 1000);
  }
  return () => {
    listeners.delete(listener);
    if (!listeners.size) {clearInterval(timer); timer = undefined;}
  };
}
const idle = () => () => {};

/** 有待确认项才订阅同一时钟；无后台请求，最后一个订阅卸载即停止。 */
export function useConfirmationTime(active: boolean) {
  return useSyncExternalStore(active ? subscribe : idle, snapshot, serverSnapshot);
}
